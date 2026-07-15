"""Format-agnostic, LLM-first raw trace ingestion source.

Deterministic code handles ONLY:
  - File discovery, grouping, and byte limits
  - Secret redaction (including array-style headers)
  - Stable artifact IDs
  - Pydantic schema validation
  - Evidence-ID verification (every cited artifact must exist)
  - Tool call/result pairing and usage deduplication
  - Coverage tracking

The LLM extracts ALL facts — timestamps, tokens, model, session identity,
messages, tool calls, HTTP status, stop reason — and cites the artifact_id
and locator for each. No provider-specific field parsing in Python.
"""

from __future__ import annotations

import glob
import hashlib
import json
import mimetypes
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from loop_engine.identifiers import compound_id
from loop_engine.models import (
    CanonicalEvent,
    CanonicalEventCandidate,
    NormalizedTraceBundle,
    RawRecordEnvelope,
)
from loop_engine.providers.base import ProviderAdapter, ProviderResponse
from loop_engine.providers.registry import (
    build_provider,
    parse_structured_response,
    resolve_model,
)
from loop_engine.security import redact_text
from loop_engine.sources.claude_normalization import _timestamp, finalize_candidates

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

_SENSITIVE_HEADER_NAMES = re.compile(
    r"(?i)^(authorization|x-api-key|cookie|set-cookie|"
    r"proxy-authorization|x-auth-token|x-session-id|"
    r"x-csrf-token|x-forwarded-authorization)$"
)

_MAX_REPAIR_ATTEMPTS = 2

_NORMALIZATION_PROMPT = """\
You are normalizing raw API trace artifacts from a proxy capture bundle.
Each artifact is a file from the same API request/response cycle: metadata
files, request bodies (with headers, tools, messages), streamed responses,
tool results, logs, or plain text.

Return a single NormalizedTraceBundle extracting ALL observable facts.

RULES:
- Every extracted fact MUST cite artifact_id and locator (JSON path, line
  number, stream event index, or null for whole-artifact).
- artifact_ids are provided in the input; use them exactly.
- Extract identity (session_id, request_id, parent_id) from whatever
  fields are present — capture_id, sessionId, uuid, etc.
- Extract timing (start_timestamp, end_timestamp, latency_ms) from
  timestamps, duration fields, or stream event ordering.
- Extract HTTP status, stop_reason, and model from metadata or response.
- Extract token usage (input_tokens, output_tokens, cache tokens) from
  usage objects wherever they appear. Do NOT double-count usage that
  appears in both request and response.
- Extract messages, tool_calls, and tool_results as separate items.
- For tool calls: extract tool_call_id, tool_name, arguments as a JSON
  string. Derive mcp_server from tool names like mcp__<server>__<method>.
- For tool results: extract tool_call_id, content, is_error.
- List pending_tool_calls: tool_call_ids with a call but no result.
- Report coverage: which artifacts you used, which you skipped, and
  which fields you could not map.
- Do not invent data absent from the artifacts.
- Serialize nested objects as JSON strings where the schema requires str.
"""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


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

    def to_llm_payload(
        self, max_chars: int, redact: bool = True
    ) -> dict[str, Any]:
        """Prepare artifact for LLM — redacted, bounded, no paths."""
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
    unresolved_fields: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure helpers (no provider knowledge)
# ---------------------------------------------------------------------------


def _detect_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _MEDIA_TYPE_MAP:
        return _MEDIA_TYPE_MAP[suffix]
    guess, _ = mimetypes.guess_type(str(path))
    return guess or "application/octet-stream"


def _is_binary(path: Path) -> bool:
    return path.suffix.lower() in _BINARY_EXTENSIONS


def _stable_artifact_id(
    source_id: str, bundle_id: str, filename: str
) -> str:
    raw = compound_id(source_id, bundle_id, filename)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _redact_artifact_content(
    text: str, media_type: str, max_chars: int
) -> str:
    """Redact credentials including array-style headers."""
    redacted = redact_text(text, max_chars) or ""

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
    path = Path(path_pattern)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file())
    return sorted(
        Path(p)
        for p in glob.glob(path_pattern, recursive=True)
        if Path(p).is_file()
    )


def _group_into_bundles(
    files: list[Path], base_path: Path | None
) -> dict[str, list[Path]]:
    bundles: dict[str, list[Path]] = {}
    for file_path in files:
        if base_path is None:
            bundle_id = file_path.stem
        elif file_path.parent == base_path:
            bundle_id = base_path.name
        else:
            try:
                relative = file_path.parent.relative_to(base_path)
                bundle_id = (
                    str(relative).replace("/", "_").replace("\\", "_")
                )
            except ValueError:
                bundle_id = file_path.parent.name
        bundles.setdefault(bundle_id, []).append(file_path)
    return bundles


# ---------------------------------------------------------------------------
# Validation (deterministic, no LLM)
# ---------------------------------------------------------------------------


def validate_bundle(
    bundle_result: NormalizedTraceBundle,
    valid_artifact_ids: set[str],
) -> list[str]:
    """Validate LLM output. Returns list of error descriptions."""
    errors: list[str] = []

    def _check_refs(
        refs: list[Any], context: str
    ) -> None:
        for ref in refs:
            if ref.artifact_id not in valid_artifact_ids:
                errors.append(
                    f"{context}: unknown artifact_id '{ref.artifact_id}'"
                )

    # Identity
    _check_refs(bundle_result.identity.evidence, "identity")
    # Timing
    _check_refs(bundle_result.timing.evidence, "timing")
    if bundle_result.timing.start_timestamp:
        try:
            _timestamp(bundle_result.timing.start_timestamp)
        except ValueError:
            errors.append(
                f"timing: invalid start_timestamp "
                f"'{bundle_result.timing.start_timestamp}'"
            )
    # HTTP
    _check_refs(bundle_result.http.evidence, "http")
    # Usage
    _check_refs(bundle_result.usage.evidence, "usage")
    # Messages
    for i, msg in enumerate(bundle_result.messages):
        _check_refs(msg.evidence, f"messages[{i}]")
    # Tool calls
    call_ids: set[str] = set()
    for i, call in enumerate(bundle_result.tool_calls):
        _check_refs(call.evidence, f"tool_calls[{i}]")
        if call.tool_call_id in call_ids:
            errors.append(
                f"tool_calls[{i}]: duplicate tool_call_id "
                f"'{call.tool_call_id}'"
            )
        call_ids.add(call.tool_call_id)
    # Tool results
    result_ids: set[str] = set()
    for i, result in enumerate(bundle_result.tool_results):
        _check_refs(result.evidence, f"tool_results[{i}]")
        if result.tool_call_id in result_ids:
            errors.append(
                f"tool_results[{i}]: duplicate tool_call_id "
                f"'{result.tool_call_id}'"
            )
        result_ids.add(result.tool_call_id)
    # Pending tool calls: must be in call_ids but not result_ids
    for pending_id in bundle_result.pending_tool_calls:
        if pending_id not in call_ids:
            errors.append(
                f"pending_tool_calls: '{pending_id}' not in tool_calls"
            )
    # Coverage
    for aid in bundle_result.coverage.artifacts_used:
        if aid not in valid_artifact_ids:
            errors.append(
                f"coverage.artifacts_used: unknown artifact_id '{aid}'"
            )

    return errors


# ---------------------------------------------------------------------------
# Bundle → CanonicalEvent conversion (deterministic)
# ---------------------------------------------------------------------------


def _bundle_to_candidates(
    bundle_result: NormalizedTraceBundle,
    artifacts: list[RawArtifactEnvelope],
) -> tuple[list[CanonicalEventCandidate], list[RawRecordEnvelope]]:
    """Convert a validated NormalizedTraceBundle into candidates + envelopes.

    All provider-specific parsing has already been done by the LLM.
    This function only:
    - Assembles CanonicalEventCandidate from the LLM's structured output
    - Builds RawRecordEnvelopes for finalize_candidates
    - Deduplicates usage across events from the same bundle
    """
    candidates: list[CanonicalEventCandidate] = []
    envelopes: list[RawRecordEnvelope] = []
    artifacts_by_id = {a.artifact_id: a for a in artifacts}

    # Shared envelope facts from the LLM's extraction
    timestamp = (
        bundle_result.timing.start_timestamp or "1970-01-01T00:00:00Z"
    )
    session_hint = bundle_result.identity.session_id
    message_id = bundle_result.identity.request_id
    model = bundle_result.http.model
    input_tokens = bundle_result.usage.input_tokens
    output_tokens = bundle_result.usage.output_tokens

    block_index = 0

    def _make(
        record_id: str,
        event_type: Literal["message", "tool_use", "tool_result", "api_error"],
        raw_ref: str,
        *,
        role: str | None = None,
        content: str | None = None,
        tool_name: str | None = None,
        tool_arguments_json: str | None = None,
        tool_result: str | None = None,
        tool_call_id: str | None = None,
        status: str | None = None,
        mcp_server: str | None = None,
        plugin_name: str | None = None,
        attribution_skill: str | None = None,
        assign_usage: bool = False,
    ) -> None:
        nonlocal block_index
        candidates.append(CanonicalEventCandidate(
            record_id=record_id,
            block_index=block_index,
            timestamp=timestamp,
            event_type=event_type,
            session_hint=session_hint,
            model=model,
            message_id=message_id,
            input_tokens=input_tokens if assign_usage else None,
            output_tokens=output_tokens if assign_usage else None,
            role=role,
            content=content,
            tool_name=tool_name,
            tool_arguments_json=tool_arguments_json,
            tool_result=tool_result,
            tool_call_id=tool_call_id,
            status=status,  # type: ignore[arg-type]
            mcp_server=mcp_server,
            plugin_name=plugin_name,
            attribution_skill=attribution_skill,
        ))
        envelopes.append(RawRecordEnvelope(
            source_id=artifacts[0].source_id,
            record_id=record_id,
            raw_ref=raw_ref,
            line_number=1,
            raw=None,
        ))
        block_index += 1

    def _ref_to_raw_ref(evidence: list[Any]) -> str:
        if evidence:
            aid = evidence[0].artifact_id
            artifact = artifacts_by_id.get(aid)
            if artifact:
                return artifact.raw_ref
        return f"raw_trace:{artifacts[0].bundle_id}"

    # Messages — assign usage to first assistant message only
    usage_assigned = False
    for i, msg in enumerate(bundle_result.messages):
        assign = not usage_assigned and msg.role == "assistant"
        if assign:
            usage_assigned = True
        _make(
            record_id=f"{artifacts[0].bundle_id}__msg_{i}",
            event_type="message",
            raw_ref=_ref_to_raw_ref(msg.evidence),
            role=msg.role,
            content=msg.content,
            assign_usage=assign,
        )

    # Tool calls
    for i, call in enumerate(bundle_result.tool_calls):
        assign = not usage_assigned
        if assign:
            usage_assigned = True
        _make(
            record_id=f"{artifacts[0].bundle_id}__call_{i}",
            event_type="tool_use",
            raw_ref=_ref_to_raw_ref(call.evidence),
            role="assistant",
            content=call.arguments_json,
            tool_name=call.tool_name,
            tool_arguments_json=call.arguments_json,
            tool_call_id=call.tool_call_id,
            mcp_server=call.mcp_server,
            plugin_name=call.plugin_name,
            attribution_skill=call.attribution_skill,
            assign_usage=assign,
        )

    # Tool results
    for i, result in enumerate(bundle_result.tool_results):
        _make(
            record_id=f"{artifacts[0].bundle_id}__result_{i}",
            event_type="tool_result",
            raw_ref=_ref_to_raw_ref(result.evidence),
            role="tool",
            content=result.content,
            tool_result=result.content,
            tool_call_id=result.tool_call_id,
            status="error" if result.is_error else "success",
        )

    return candidates, envelopes


# ---------------------------------------------------------------------------
# RawTraceSource
# ---------------------------------------------------------------------------


class RawTraceSource:
    """Format-agnostic, LLM-first raw trace ingestion source.

    The LLM extracts ALL facts from arbitrary text-based trace artifacts
    and returns a NormalizedTraceBundle with evidence citations.
    Deterministic code validates, pairs, deduplicates, and converts.
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
        max_output_tokens: int = 16_384,
        redact_before_egress: bool = True,
        provider_name: str = "anthropic",
        repair: bool = True,
        *,
        provider: ProviderAdapter | None = None,
    ) -> None:
        if max_artifact_bytes < 1:
            raise ValueError("max_artifact_bytes must be >= 1")
        if max_bundle_bytes < max_artifact_bytes:
            raise ValueError(
                "max_bundle_bytes must be >= max_artifact_bytes"
            )
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

    # -- File discovery (unchanged) --

    def iter_bundles(self) -> Iterable[list[RawArtifactEnvelope]]:
        files = _discover_files(self.path_pattern)
        path = Path(self.path_pattern)
        base_path = (
            path
            if path.is_dir()
            else path.parent if path.is_file() else None
        )
        grouped = _group_into_bundles(files, base_path)
        total_bytes = 0

        for bundle_id, bundle_files in sorted(grouped.items()):
            bundle: list[RawArtifactEnvelope] = []
            bundle_bytes = 0
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
                            f"({file_bytes} > "
                            f"{self.max_artifact_bytes})"
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

    # -- Payload building --

    def _build_payload(
        self, artifacts: list[RawArtifactEnvelope]
    ) -> str:
        items = [
            a.to_llm_payload(
                self.max_artifact_chars,
                redact=self.redact_before_egress,
            )
            for a in artifacts
        ]
        payload = json.dumps(
            {"bundle_id": artifacts[0].bundle_id, "artifacts": items},
            ensure_ascii=False,
        )
        if len(payload) > self.max_input_chars:
            raise RuntimeError(
                f"raw_trace bundle payload exceeds input limit "
                f"({len(payload)} > {self.max_input_chars} chars)"
            )
        return payload

    # -- Bounded validation loop --

    def _normalize_bundle(
        self,
        bundle: list[RawArtifactEnvelope],
    ) -> NormalizedTraceBundle:
        """Normalize a bundle with up to _MAX_REPAIR_ATTEMPTS retries.

        1. Send artifacts to LLM → get NormalizedTraceBundle.
        2. Validate schema + evidence citations.
        3. If errors, send errors + omitted facts back to LLM.
        4. Repeat up to _MAX_REPAIR_ATTEMPTS times.
        """
        payload = self._build_payload(bundle)
        schema = NormalizedTraceBundle.model_json_schema()
        valid_ids = {a.artifact_id for a in bundle}

        response = self._provider.request_structured(
            model=resolve_model(self.model),
            system_prompt=_NORMALIZATION_PROMPT,
            payload=payload,
            schema=schema,
            max_output_tokens=self.max_output_tokens,
            operation="raw_trace normalization",
        )

        for attempt in range(_MAX_REPAIR_ATTEMPTS + 1):
            try:
                result = parse_structured_response(
                    response,
                    NormalizedTraceBundle,
                    "raw_trace normalization",
                )
            except (RuntimeError, Exception) as parse_error:
                if attempt >= _MAX_REPAIR_ATTEMPTS:
                    raise
                response = self._request_repair(
                    payload, schema, response.raw_text,
                    [str(parse_error)],
                )
                continue

            assert isinstance(result, NormalizedTraceBundle)
            errors = validate_bundle(result, valid_ids)
            if not errors:
                return result
            if attempt >= _MAX_REPAIR_ATTEMPTS:
                raise RuntimeError(
                    f"raw_trace normalization failed validation "
                    f"after {_MAX_REPAIR_ATTEMPTS} repair attempts: "
                    + "; ".join(errors)
                )
            response = self._request_repair(
                payload, schema, response.raw_text, errors,
            )

        # Unreachable but satisfies type checker
        raise RuntimeError("raw_trace normalization loop exhausted")

    def _request_repair(
        self,
        original_payload: str,
        schema: dict[str, Any],
        previous_response: str,
        errors: list[str],
    ) -> ProviderResponse:
        """Send a repair request with the errors and schema."""
        repair_prompt = (
            "Your previous response failed validation.\n\n"
            "Errors:\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n\n"
            f"Your response (first 3000 chars):\n"
            f"{previous_response[:3000]}\n\n"
            f"Target JSON schema:\n"
            f"{json.dumps(schema, indent=2)[:3000]}\n\n"
            "Return a corrected NormalizedTraceBundle. "
            "Fix the cited errors. "
            "Every fact must cite an artifact_id from the input."
        )
        return self._provider.request_structured(
            model=resolve_model(self.model),
            system_prompt=_NORMALIZATION_PROMPT,
            payload=repair_prompt,
            schema=schema,
            max_output_tokens=self.max_output_tokens,
            operation="raw_trace normalization (repair)",
        )

    # -- Main pipeline --

    def iter_events(self) -> Iterable[CanonicalEvent]:
        all_events: list[CanonicalEvent] = []

        for bundle in self.iter_bundles():
            result = self._normalize_bundle(bundle)

            # Update coverage from LLM's self-report
            self.coverage.normalized_artifacts += len(
                result.coverage.artifacts_used
            )
            self.coverage.unresolved_fields.extend(
                result.coverage.unresolved_fields
            )

            # Convert to candidates deterministically
            candidates, envelopes = _bundle_to_candidates(result, bundle)
            self.coverage.total_events += len(candidates)

            # Finalize: IDs, pairing, dedup
            events = finalize_candidates(envelopes, candidates)
            all_events.extend(events)

        return all_events
