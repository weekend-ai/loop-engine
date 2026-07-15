from __future__ import annotations

import glob
import hashlib
import json
import mimetypes
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loop_engine.identifiers import compound_id
from loop_engine.models import (
    CanonicalEvent,
    CanonicalEventCandidate,
    LlmNormalizationBatch,
    LlmNormalizationCandidate,
    RawRecordEnvelope,
)
from loop_engine.providers.base import ProviderAdapter
from loop_engine.providers.registry import (
    build_provider,
    request_and_validate,
    resolve_model,
)
from loop_engine.security import redact_text
from loop_engine.sources.claude_normalization import finalize_candidates

_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".avi", ".mkv",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".pyo",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".sqlite", ".db", ".wasm",
})

_MEDIA_TYPE_MAP: dict[str, str] = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".xml": "application/xml",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
}

# Headers that carry secrets when represented as 2-element arrays
_SENSITIVE_HEADER_NAMES = re.compile(
    r"(?i)^(authorization|x-api-key|cookie|set-cookie|proxy-authorization|"
    r"x-auth-token|x-session-id|x-csrf-token|x-forwarded-authorization)$"
)

_RAW_TRACE_PROMPT = """You are normalizing raw API trace artifacts from a proxy capture bundle.
Each artifact is a file from the same API request/response cycle. Together they represent one or
more AI agent interactions: metadata, request bodies with tools, streamed responses, and tool
results.

Correlate the artifacts to reconstruct:
- Individual messages, tool calls, and tool results as separate events
- Tool call/result pairing via tool_call_id
- MCP server attribution from tool names (mcp__<server>__<method>)
- Plugin and skill attribution if present
- Event ordering from timestamps, sequence numbers, or stream position

Rules:
- Each event must reference exactly one artifact_id (the artifact it was extracted from)
- Use record_id = artifact_id for single-event artifacts
- For multi-event artifacts (e.g. a response stream with text + tool_use blocks), append
  the block index: record_id = "<artifact_id>__<block_index>"
- event_type must be one of: message, tool_use, tool_result, api_error
- tool_result events MUST have role='tool'
- tool_use events MUST have tool_name
- Serialize tool arguments as a JSON string in tool_arguments_json, not as a nested object
- Do not invent data not present in the artifacts
- Do not repeat metadata/envelope facts (timestamp, tokens, model) — those are joined locally
"""


@dataclass
class RawArtifactEnvelope:
    """Bounded container for one file in a raw trace bundle."""

    source_id: str
    artifact_id: str
    bundle_id: str
    filename: str
    media_type: str
    sequence: int
    content: str
    raw_ref: str
    byte_size: int

    def to_llm_payload(self, max_chars: int, redact: bool = True) -> dict[str, Any]:
        """Prepare artifact for LLM consumption — redacted, bounded, no paths."""
        text = self.content
        if redact:
            text = _redact_artifact_content(text, self.media_type, max_chars)
        else:
            text = redact_text(text, max_chars) or ""
        return {
            "artifact_id": self.artifact_id,
            "filename": self.filename,
            "media_type": self.media_type,
            "sequence": self.sequence,
            "content": text,
        }


@dataclass
class IngestionCoverage:
    """Track normalization coverage for reporting."""

    total_artifacts: int = 0
    normalized_artifacts: int = 0
    skipped_artifacts: list[dict[str, str]] = field(default_factory=list)
    total_events: int = 0


def _detect_media_type(path: Path) -> str:
    """Detect media type from extension, with fallback to mimetypes."""
    suffix = path.suffix.lower()
    if suffix in _MEDIA_TYPE_MAP:
        return _MEDIA_TYPE_MAP[suffix]
    guess, _ = mimetypes.guess_type(str(path))
    return guess or "application/octet-stream"


def _is_binary(path: Path) -> bool:
    """Check if a file is binary based on extension."""
    return path.suffix.lower() in _BINARY_EXTENSIONS


def _stable_artifact_id(source_id: str, bundle_id: str, filename: str) -> str:
    """Generate a stable, deterministic artifact ID."""
    raw = compound_id(source_id, bundle_id, filename)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _redact_artifact_content(
    text: str, media_type: str, max_chars: int
) -> str:
    """Redact credentials from artifact content.

    Handles both standard key=value patterns and array-style headers
    like ["Authorization", "Bearer ..."].
    """
    # First pass: standard redaction
    redacted = redact_text(text, max_chars) or ""

    # Second pass: array-style header redaction ["HeaderName", "value"]
    # This catches patterns like ["Authorization", "Bearer sk-ant-..."]
    def _redact_header_array(match: re.Match[str]) -> str:
        name = match.group(1)
        if _SENSITIVE_HEADER_NAMES.match(name):
            return f'["{name}", "[REDACTED]"]'
        return match.group(0)

    redacted = re.sub(
        r'\["([^"]+)",\s*"([^"]*)"(?:,\s*"[^"]*")*\]',
        _redact_header_array,
        redacted,
    )
    return redacted


def _discover_files(path_pattern: str) -> list[Path]:
    """Discover files from a path, directory, or glob pattern."""
    path = Path(path_pattern)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file())
    # Glob pattern
    return sorted(Path(p) for p in glob.glob(path_pattern, recursive=True) if Path(p).is_file())


def _group_into_bundles(files: list[Path], base_path: Path | None) -> dict[str, list[Path]]:
    """Group files into bundles by their parent directory.

    Files in the same immediate subdirectory are assumed to be part of
    the same request/response bundle. If base_path is the direct parent
    of all files, they form a single bundle named after that directory.
    """
    bundles: dict[str, list[Path]] = {}
    for file_path in files:
        if base_path is None:
            bundle_id = file_path.stem
        elif file_path.parent == base_path:
            # All files directly in the target directory = one bundle
            bundle_id = base_path.name
        else:
            # Subdirectory = bundle name
            try:
                relative = file_path.parent.relative_to(base_path)
                bundle_id = str(relative).replace("/", "_").replace("\\", "_")
            except ValueError:
                bundle_id = file_path.parent.name
        bundles.setdefault(bundle_id, []).append(file_path)
    return bundles


class RawTraceSource:
    """Format-agnostic raw trace ingestion source.

    Accepts arbitrary text-based trace artifacts (JSON, JSONL, SSE, logs,
    plain text) and uses LLM normalization to produce canonical events.

    Files in the same directory are bundled together so the LLM can
    correlate metadata, request, streamed response, and tool results.
    """

    def __init__(
        self,
        source_id: str,
        path_pattern: str,
        model: str = "sonnet",
        timeout_seconds: int = 120,
        max_artifact_bytes: int = 1 * 1024 * 1024,
        max_bundle_bytes: int = 5 * 1024 * 1024,
        max_total_bytes: int = 50 * 1024 * 1024,
        max_input_chars: int = 100_000,
        max_artifact_chars: int = 10_000,
        max_output_tokens: int = 8_192,
        redact_before_egress: bool = True,
        provider_name: str = "anthropic",
        repair: bool = True,
        *,
        provider: ProviderAdapter | None = None,
    ) -> None:
        if max_artifact_bytes < 1:
            raise ValueError("max_artifact_bytes must be >= 1")
        if max_bundle_bytes < max_artifact_bytes:
            raise ValueError("max_bundle_bytes must be >= max_artifact_bytes")

        self.source_id = source_id
        self.path_pattern = path_pattern
        self.model = model
        self.max_artifact_bytes = max_artifact_bytes
        self.max_bundle_bytes = max_bundle_bytes
        self.max_total_bytes = max_total_bytes
        self.max_input_chars = max_input_chars
        self.max_artifact_chars = max_artifact_chars
        self.max_output_tokens = max_output_tokens
        self.redact_before_egress = redact_before_egress
        self.repair = repair
        self._provider = provider or build_provider(
            provider_name,  # type: ignore[arg-type]
            timeout_seconds=timeout_seconds,
        )
        self.coverage = IngestionCoverage()

    def iter_bundles(self) -> Iterable[list[RawArtifactEnvelope]]:
        """Discover, validate, and yield artifact bundles."""
        files = _discover_files(self.path_pattern)
        path = Path(self.path_pattern)
        base_path = path if path.is_dir() else path.parent if path.is_file() else None
        grouped = _group_into_bundles(files, base_path)
        total_bytes = 0

        for bundle_id, bundle_files in sorted(grouped.items()):
            bundle: list[RawArtifactEnvelope] = []
            bundle_bytes = 0

            # Sort files for stable ordering: metadata first, then alphabetical
            sorted_files = sorted(bundle_files, key=lambda p: (
                0 if "metadata" in p.stem.lower() else
                1 if "request" in p.stem.lower() else
                2 if "response" in p.stem.lower() else
                3,
                p.name,
            ))

            for sequence, file_path in enumerate(sorted_files):
                if _is_binary(file_path):
                    self.coverage.skipped_artifacts.append({
                        "file": str(file_path),
                        "reason": "binary file",
                    })
                    continue

                file_bytes = file_path.stat().st_size
                if file_bytes > self.max_artifact_bytes:
                    self.coverage.skipped_artifacts.append({
                        "file": str(file_path),
                        "reason": (
                            f"exceeds artifact limit "
                            f"({file_bytes} > {self.max_artifact_bytes})"
                        ),
                    })
                    continue

                bundle_bytes += file_bytes
                if bundle_bytes > self.max_bundle_bytes:
                    self.coverage.skipped_artifacts.append({
                        "file": str(file_path),
                        "reason": "bundle byte limit exceeded",
                    })
                    continue

                total_bytes += file_bytes
                if total_bytes > self.max_total_bytes:
                    raise ValueError(
                        f"raw_trace total byte limit exceeded "
                        f"({total_bytes} > {self.max_total_bytes})"
                    )

                try:
                    content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    self.coverage.skipped_artifacts.append({
                        "file": str(file_path),
                        "reason": "not valid UTF-8",
                    })
                    continue

                # Relative filename for the artifact (no absolute paths)
                try:
                    relative_name = str(file_path.relative_to(
                        base_path if base_path else file_path.parent
                    ))
                except ValueError:
                    relative_name = file_path.name

                artifact_id = _stable_artifact_id(
                    self.source_id, bundle_id, relative_name
                )
                media_type = _detect_media_type(file_path)
                raw_ref = f"file://{file_path.resolve()}"

                self.coverage.total_artifacts += 1
                bundle.append(RawArtifactEnvelope(
                    source_id=self.source_id,
                    artifact_id=artifact_id,
                    bundle_id=bundle_id,
                    filename=relative_name,
                    media_type=media_type,
                    sequence=sequence,
                    content=content,
                    raw_ref=raw_ref,
                    byte_size=file_bytes,
                ))

            if bundle:
                yield bundle

    def _build_payload(
        self, bundle: list[RawArtifactEnvelope]
    ) -> str:
        """Build the LLM payload from a bundle of artifacts."""
        artifacts = [
            artifact.to_llm_payload(
                self.max_artifact_chars,
                redact=self.redact_before_egress,
            )
            for artifact in bundle
        ]
        payload = json.dumps(
            {"bundle_id": bundle[0].bundle_id, "artifacts": artifacts},
            ensure_ascii=False,
        )
        if len(payload) > self.max_input_chars:
            raise RuntimeError(
                f"raw_trace bundle payload exceeds input limit "
                f"({len(payload)} > {self.max_input_chars} chars)"
            )
        return payload

    def _enrich_from_artifacts(
        self,
        llm_candidate: LlmNormalizationCandidate,
        artifacts_by_id: dict[str, RawArtifactEnvelope],
    ) -> tuple[CanonicalEventCandidate, RawRecordEnvelope]:
        """Join LLM-interpreted fields with deterministic artifact facts.

        Returns both the enriched candidate and the corresponding
        RawRecordEnvelope for finalization.
        """
        # The record_id may be "artifact_id" or "artifact_id__block_index"
        base_artifact_id = llm_candidate.record_id.split("__")[0]
        artifact = artifacts_by_id.get(base_artifact_id)

        # Extract envelope-level facts from artifact content
        timestamp = None
        model_hint = None
        input_tokens = None
        output_tokens = None
        session_hint = None
        message_id = None

        if artifact:
            # Try to extract metadata from the bundle's metadata artifact
            metadata_artifact = next(
                (a for a in artifacts_by_id.values()
                 if "metadata" in a.filename.lower()),
                None,
            )
            if metadata_artifact:
                try:
                    meta = json.loads(metadata_artifact.content)
                    timestamp = meta.get("timestamp")
                    model_hint = meta.get("model")
                    session_hint = meta.get("capture_id") or meta.get("session_id")
                    message_id = meta.get("request_id")
                except (json.JSONDecodeError, AttributeError):
                    pass

            # Try to extract usage from response stream
            response_artifact = next(
                (a for a in artifacts_by_id.values()
                 if "response" in a.filename.lower()),
                None,
            )
            if response_artifact:
                for line in response_artifact.content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("type") == "message_start":
                            msg = record.get("message", {})
                            usage = msg.get("usage", {})
                            input_tokens = usage.get("input_tokens")
                            if not model_hint:
                                model_hint = msg.get("model")
                            if not message_id:
                                message_id = msg.get("id")
                        elif record.get("type") == "message_delta":
                            usage = record.get("usage", {})
                            output_tokens = usage.get("output_tokens")
                    except (json.JSONDecodeError, AttributeError):
                        continue

        # Build the envelope for finalization
        envelope = RawRecordEnvelope(
            source_id=self.source_id,
            record_id=llm_candidate.record_id,
            raw_ref=artifact.raw_ref if artifact else f"raw_trace:{base_artifact_id}",
            line_number=1,
            raw=None,
        )

        candidate = CanonicalEventCandidate(
            record_id=llm_candidate.record_id,
            block_index=llm_candidate.block_index,
            timestamp=timestamp or "1970-01-01T00:00:00Z",
            session_hint=session_hint,
            model=model_hint,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            message_id=message_id,
            # LLM-interpreted fields
            event_type=llm_candidate.event_type,
            role=llm_candidate.role,
            content=llm_candidate.content,
            tool_name=llm_candidate.tool_name,
            tool_arguments_json=llm_candidate.tool_arguments_json,
            tool_result=llm_candidate.tool_result,
            tool_call_id=llm_candidate.tool_call_id,
            status=llm_candidate.status,
            mcp_server=llm_candidate.mcp_server,
            plugin_name=llm_candidate.plugin_name,
            attribution_skill=llm_candidate.attribution_skill,
        )

        return candidate, envelope

    def iter_events(self) -> Iterable[CanonicalEvent]:
        """Ingest all bundles and produce canonical events."""
        all_events: list[CanonicalEvent] = []

        for bundle in self.iter_bundles():
            artifacts_by_id = {a.artifact_id: a for a in bundle}
            payload = self._build_payload(bundle)

            result = request_and_validate(
                self._provider,
                model=resolve_model(self.model),
                system_prompt=_RAW_TRACE_PROMPT,
                payload=payload,
                target_type=LlmNormalizationBatch,
                max_output_tokens=self.max_output_tokens,
                operation="raw_trace normalization",
                repair=self.repair,
            )
            assert isinstance(result, LlmNormalizationBatch)

            candidates: list[CanonicalEventCandidate] = []
            envelopes: list[RawRecordEnvelope] = []

            for llm_candidate in result.events:
                candidate, envelope = self._enrich_from_artifacts(
                    llm_candidate, artifacts_by_id
                )
                candidates.append(candidate)
                envelopes.append(envelope)

            self.coverage.normalized_artifacts += len(
                {c.record_id.split("__")[0] for c in candidates}
            )
            self.coverage.total_events += len(candidates)

            events = finalize_candidates(envelopes, candidates)
            all_events.extend(events)

        return all_events
