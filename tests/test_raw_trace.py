from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from loop_engine.config import load_config
from loop_engine.providers.base import ProviderResponse
from loop_engine.sources.raw_trace import (
    RawTraceSource,
    _detect_media_type,
    _is_binary,
    _redact_artifact_content,
)

FIXTURE_DIR = Path("tests/fixtures/relay_capture/20260714T172453.744313-fd84d2c9")


def _llm_candidate(**overrides: Any) -> dict[str, Any]:
    """Build a complete LlmNormalizationCandidate dict with all required fields."""
    base: dict[str, Any] = {
        "record_id": "artifact",
        "block_index": 0,
        "event_type": "message",
        "role": None,
        "content": None,
        "tool_name": None,
        "tool_arguments_json": None,
        "tool_result": None,
        "tool_call_id": None,
        "status": None,
        "mcp_server": None,
        "plugin_name": None,
        "attribution_skill": None,
    }
    base.update(overrides)
    return base


class FakeProvider:
    """Test provider returning canned responses."""

    def __init__(self, response: dict[str, Any] | str | Exception) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def request_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        payload: str,
        schema: dict[str, Any],
        max_output_tokens: int,
        operation: str,
    ) -> ProviderResponse:
        self.calls.append({
            "model": model,
            "system": system_prompt,
            "payload": payload,
            "schema": schema,
            "max_output_tokens": max_output_tokens,
            "operation": operation,
        })
        if isinstance(self._response, Exception):
            raise self._response
        if isinstance(self._response, str):
            return ProviderResponse(raw_text=self._response)
        return ProviderResponse(raw_text=json.dumps(self._response))


# ---------------------------------------------------------------------------
# File discovery and bundling
# ---------------------------------------------------------------------------

def test_raw_trace_discovers_bundle_from_directory() -> None:
    source = RawTraceSource(
        "test", str(FIXTURE_DIR), provider=FakeProvider({"events": []})
    )
    bundles = list(source.iter_bundles())
    assert len(bundles) == 1
    bundle = bundles[0]
    assert len(bundle) == 4
    filenames = {a.filename for a in bundle}
    assert "metadata.json" in filenames or any("metadata" in f for f in filenames)


def test_raw_trace_stable_artifact_ids() -> None:
    """Artifact IDs are deterministic across runs."""
    source = RawTraceSource(
        "test", str(FIXTURE_DIR), provider=FakeProvider({"events": []})
    )
    ids_1 = [a.artifact_id for bundle in source.iter_bundles() for a in bundle]
    source2 = RawTraceSource(
        "test", str(FIXTURE_DIR), provider=FakeProvider({"events": []})
    )
    ids_2 = [a.artifact_id for bundle in source2.iter_bundles() for a in bundle]
    assert ids_1 == ids_2
    assert len(set(ids_1)) == 4  # all unique


def test_raw_trace_media_type_detection() -> None:
    assert _detect_media_type(Path("test.json")) == "application/json"
    assert _detect_media_type(Path("stream.jsonl")) == "application/x-ndjson"
    assert _detect_media_type(Path("notes.txt")) == "text/plain"
    assert _detect_media_type(Path("data.csv")) == "text/csv"


def test_raw_trace_rejects_binary_files() -> None:
    assert _is_binary(Path("image.png"))
    assert _is_binary(Path("archive.zip"))
    assert _is_binary(Path("model.bin"))
    assert not _is_binary(Path("data.json"))
    assert not _is_binary(Path("trace.log"))


def test_raw_trace_skips_binary_and_tracks_coverage(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text('{"test": true}')
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")

    source = RawTraceSource(
        "test", str(tmp_path), provider=FakeProvider({"events": []})
    )
    bundles = list(source.iter_bundles())
    assert len(bundles) == 1
    assert len(bundles[0]) == 1  # only the JSON
    assert len(source.coverage.skipped_artifacts) == 1
    assert source.coverage.skipped_artifacts[0]["reason"] == "binary file"


# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------

def test_raw_trace_enforces_per_artifact_limit(tmp_path: Path) -> None:
    (tmp_path / "big.json").write_text("x" * 1000)
    source = RawTraceSource(
        "test", str(tmp_path),
        max_artifact_bytes=100,
        max_bundle_bytes=10_000,
        max_total_bytes=100_000,
        provider=FakeProvider({"events": []}),
    )
    bundles = list(source.iter_bundles())
    assert len(bundles) == 0  # skipped, not failed
    assert len(source.coverage.skipped_artifacts) == 1
    assert "exceeds artifact limit" in source.coverage.skipped_artifacts[0]["reason"]


def test_raw_trace_enforces_total_limit(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"file_{i}.json").write_text('{"data": "' + "x" * 100 + '"}')
    source = RawTraceSource(
        "test", str(tmp_path),
        max_artifact_bytes=10_000,
        max_bundle_bytes=10_000,
        max_total_bytes=500,  # lower than total of all files
        provider=FakeProvider({"events": []}),
    )
    with pytest.raises(ValueError, match="total byte limit exceeded"):
        list(source.iter_bundles())


# ---------------------------------------------------------------------------
# Credential redaction
# ---------------------------------------------------------------------------

def test_redact_array_style_headers() -> None:
    """Headers as ["Authorization", "Bearer ..."] must be redacted."""
    content = json.dumps({
        "headers": [
            ["Authorization", "Bearer sk-ant-api03-FAKEtoken1234567890"],
            ["x-api-key", "sk-ant-api03-FAKEtoken1234567890"],
            ["Cookie", "session=secret_session_value_12345"],
            ["Content-Type", "application/json"],
        ]
    })
    redacted = _redact_artifact_content(content, "application/json", 10_000)
    assert "FAKEtoken" not in redacted
    assert "secret_session" not in redacted
    assert "[REDACTED]" in redacted
    # Content-Type should NOT be redacted
    assert "application/json" in redacted


def test_redact_bearer_token_in_plaintext() -> None:
    text = "Authorization: Bearer sk-ant-api03-realtoken1234567890"
    redacted = _redact_artifact_content(text, "text/plain", 10_000)
    assert "realtoken" not in redacted
    assert "[REDACTED]" in redacted


def test_fixture_auth_tokens_are_redacted() -> None:
    """The acceptance test fixture's auth tokens must never reach the LLM."""
    request_content = (FIXTURE_DIR / "request.json").read_text()
    redacted = _redact_artifact_content(request_content, "application/json", 100_000)

    # The fixture contains sk-ant-api03-... tokens
    assert "FAKE_TOKEN_DO_NOT_USE" not in redacted
    assert "sk-ant-api03" not in redacted
    # Auth header values should be redacted
    assert "Authorization" in redacted  # header name preserved
    assert "[REDACTED]" in redacted


def test_absolute_paths_never_in_payload() -> None:
    """Local absolute paths must not be sent to the LLM."""
    source = RawTraceSource(
        "test", str(FIXTURE_DIR), provider=FakeProvider({"events": []})
    )
    for bundle in source.iter_bundles():
        payload = source._build_payload(bundle)
        # No absolute paths
        assert "/root/" not in payload
        assert "/home/" not in payload
        assert "file://" not in payload
        # Relative filenames are OK
        assert "metadata.json" in payload


# ---------------------------------------------------------------------------
# LLM normalization integration (with FakeProvider)
# ---------------------------------------------------------------------------

def test_raw_trace_produces_canonical_events() -> None:
    """Full pipeline: fixture → bundle → LLM normalization → canonical events."""
    # Get artifact IDs from the fixture
    source_for_ids = RawTraceSource(
        "relay", str(FIXTURE_DIR), provider=FakeProvider({"events": []})
    )
    bundles = list(source_for_ids.iter_bundles())
    artifact_ids = {a.filename: a.artifact_id for bundle in bundles for a in bundle}

    # Build a response with the correct artifact-based record IDs
    response_id = next(
        aid for fname, aid in artifact_ids.items() if "response" in fname
    )
    tool_results_id = next(
        aid for fname, aid in artifact_ids.items() if "tool_results" in fname
    )

    llm_response = {
        "events": [
            _llm_candidate(
                record_id=f"{response_id}__0",
                block_index=0,
                event_type="message",
                role="assistant",
                content="I'll search DataHub for the customer_events dataset.",
            ),
            _llm_candidate(
                record_id=f"{response_id}__1",
                block_index=1,
                event_type="tool_use",
                role="assistant",
                tool_name="mcp__datahub__search_datasets",
                tool_arguments_json='{"query": "customer_events", "platform": "snowflake"}',
                tool_call_id="toolu_01search",
                mcp_server="datahub",
            ),
            _llm_candidate(
                record_id=f"{response_id}__2",
                block_index=2,
                event_type="tool_use",
                role="assistant",
                tool_name="mcp__datahub__get_dataset_schema",
                tool_arguments_json='{"dataset_urn": "urn:li:dataset:(...,PROD)"}',
                tool_call_id="toolu_02schema",
                mcp_server="datahub",
            ),
            _llm_candidate(
                record_id=f"{tool_results_id}__0",
                block_index=0,
                event_type="tool_result",
                role="tool",
                tool_call_id="toolu_01search",
                tool_result="[{...customer_events...}]",
                status="success",
                content="[{...customer_events...}]",
            ),
            _llm_candidate(
                record_id=f"{tool_results_id}__1",
                block_index=1,
                event_type="tool_result",
                role="tool",
                tool_call_id="toolu_02schema",
                tool_result="{fields: [...]}",
                status="success",
                content="{fields: [...]}",
            ),
        ]
    }
    fake_provider = FakeProvider(llm_response)
    source = RawTraceSource("relay", str(FIXTURE_DIR), provider=fake_provider, repair=False)

    events = list(source.iter_events())

    assert len(events) == 5
    # Verify tool call/result pairing
    tool_uses = [e for e in events if e.event_type == "tool_use"]
    tool_results = [e for e in events if e.event_type == "tool_result"]
    assert len(tool_uses) == 2
    assert len(tool_results) == 2

    # MCP server attribution
    assert all(e.mcp_server == "datahub" for e in tool_uses)

    # Tool pairing via finalize_candidates
    for use in tool_uses:
        assert use.paired_event_id is not None
    for result in tool_results:
        assert result.paired_event_id is not None

    # Deterministic enrichment: metadata facts joined back
    for event in events:
        assert event.model == "claude-sonnet-4-20250514"
        assert event.input_tokens == 1247

    # Coverage tracking
    assert source.coverage.total_artifacts == 4
    assert source.coverage.normalized_artifacts >= 2
    assert source.coverage.total_events == 5

    # Verify the payload sent to the LLM has no auth tokens
    call = fake_provider.calls[0]
    payload = call["payload"]
    assert "FAKE_TOKEN_DO_NOT_USE" not in payload
    assert "sk-ant-api03" not in payload


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_raw_trace_config_requires_claude_sdk_normalizer(tmp_path: Path) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text(
        "version: 1\n"
        "sources:\n"
        "  - id: relay\n"
        "    type: raw_trace\n"
        "    path: ./captures\n"
        "    normalizer: rule_based\n"
    )
    with pytest.raises(ValidationError, match="require normalizer: claude_sdk"):
        load_config(path)


def test_raw_trace_config_requires_egress(tmp_path: Path) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text(
        "version: 1\n"
        "sources:\n"
        "  - id: relay\n"
        "    type: raw_trace\n"
        "    path: ./captures\n"
        "    normalizer: claude_sdk\n"
    )
    with pytest.raises(ValidationError, match="external_data_egress_allowed"):
        load_config(path)


def test_raw_trace_config_accepted(tmp_path: Path) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text(
        "version: 1\n"
        "sources:\n"
        "  - id: relay\n"
        "    type: raw_trace\n"
        "    path: ./captures\n"
        "    normalizer: claude_sdk\n"
        "analysis:\n"
        "  external_data_egress_allowed: true\n"
    )
    config = load_config(path)
    assert config.sources[0].type == "raw_trace"
    assert config.sources[0].normalizer == "claude_sdk"


# ---------------------------------------------------------------------------
# Bundle ordering
# ---------------------------------------------------------------------------

def test_bundle_orders_metadata_first() -> None:
    source = RawTraceSource(
        "test", str(FIXTURE_DIR), provider=FakeProvider({"events": []})
    )
    for bundle in source.iter_bundles():
        assert bundle[0].filename.startswith("metadata") or "metadata" in bundle[0].filename
        # request before response
        req_idx = next(
            (i for i, a in enumerate(bundle) if "request" in a.filename), None
        )
        resp_idx = next(
            (i for i, a in enumerate(bundle) if "response" in a.filename), None
        )
        if req_idx is not None and resp_idx is not None:
            assert req_idx < resp_idx
