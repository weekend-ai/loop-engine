"""Tests for the LLM-first raw_trace ingestion source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from loop_engine.config import load_config
from loop_engine.models import NormalizedTraceBundle
from loop_engine.providers.base import ProviderResponse
from loop_engine.sources.raw_trace import (
    RawTraceSource,
    _detect_media_type,
    _is_binary,
    _redact_artifact_content,
    validate_bundle,
)

FIXTURE_DIR = Path(
    "tests/fixtures/relay_capture/20260714T172453.744313-fd84d2c9"
)


def _evidence(artifact_id: str, locator: str | None = None) -> dict[str, Any]:
    return {"artifact_id": artifact_id, "locator": locator}


def _make_bundle_response(
    artifact_ids: dict[str, str], **overrides: Any
) -> dict[str, Any]:
    """Build a complete NormalizedTraceBundle dict for testing."""
    meta_id = next(
        aid for fname, aid in artifact_ids.items() if "metadata" in fname
    )
    req_id = next(
        aid for fname, aid in artifact_ids.items() if "request" in fname
    )
    resp_id = next(
        aid for fname, aid in artifact_ids.items() if "response" in fname
    )
    tool_id = next(
        aid for fname, aid in artifact_ids.items() if "tool_results" in fname
    )
    base: dict[str, Any] = {
        "identity": {
            "session_id": "20260714T172453.744313-fd84d2c9",
            "request_id": "req_01ABC123def456",
            "parent_id": None,
            "evidence": [_evidence(meta_id)],
        },
        "timing": {
            "start_timestamp": "2026-07-14T17:24:53.744313Z",
            "end_timestamp": None,
            "latency_ms": 8212,
            "evidence": [_evidence(meta_id, "$.duration_ms")],
        },
        "http": {
            "status_code": 200,
            "stop_reason": "tool_use",
            "model": "claude-sonnet-4-20250514",
            "evidence": [
                _evidence(meta_id, "$.http_status"),
                _evidence(resp_id, "line 11, message_delta"),
            ],
        },
        "usage": {
            "input_tokens": 14042,
            "output_tokens": 409,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
            "evidence": [
                _evidence(resp_id, "line 1, message_start.usage"),
                _evidence(resp_id, "line 11, message_delta.usage"),
            ],
        },
        "messages": [
            {
                "role": "user",
                "content": "Find me the customer_events dataset.",
                "evidence": [_evidence(req_id, "$.messages[0]")],
            },
            {
                "role": "assistant",
                "content": (
                    "I'll search DataHub for the customer_events "
                    "dataset on Snowflake."
                ),
                "evidence": [
                    _evidence(resp_id, "line 3, content_block_delta")
                ],
            },
        ],
        "tool_calls": [
            {
                "tool_call_id": "toolu_01search",
                "tool_name": "mcp__datahub__search_datasets",
                "arguments_json": (
                    '{"query": "customer_events", '
                    '"platform": "snowflake"}'
                ),
                "mcp_server": "datahub",
                "plugin_name": None,
                "attribution_skill": None,
                "evidence": [
                    _evidence(resp_id, "line 5, content_block_start")
                ],
            },
            {
                "tool_call_id": "toolu_02schema",
                "tool_name": "mcp__datahub__get_dataset_schema",
                "arguments_json": '{"dataset_urn": "urn:li:dataset:..."}',
                "mcp_server": "datahub",
                "plugin_name": None,
                "attribution_skill": None,
                "evidence": [
                    _evidence(resp_id, "line 8, content_block_start")
                ],
            },
        ],
        "tool_results": [
            {
                "tool_call_id": "toolu_01search",
                "content": '[{"name": "customer_events"}]',
                "is_error": False,
                "evidence": [
                    _evidence(tool_id, "$.tool_results[0]")
                ],
            },
            {
                "tool_call_id": "toolu_02schema",
                "content": '{"fields": [...]}',
                "is_error": False,
                "evidence": [
                    _evidence(tool_id, "$.tool_results[1]")
                ],
            },
        ],
        "pending_tool_calls": [],
        "coverage": {
            "artifacts_used": [meta_id, req_id, resp_id, tool_id],
            "artifacts_skipped": [],
            "unresolved_fields": [],
        },
    }
    base.update(overrides)
    return base


class FakeProvider:
    """Test provider returning canned responses."""

    def __init__(
        self, response: dict[str, Any] | str | Exception
    ) -> None:
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


def _get_artifact_ids() -> dict[str, str]:
    """Get stable artifact IDs from the fixture."""
    source = RawTraceSource(
        "relay", str(FIXTURE_DIR),
        provider=FakeProvider({"identity": {}}),  # unused
    )
    ids: dict[str, str] = {}
    for bundle in source.iter_bundles():
        for a in bundle:
            ids[a.filename] = a.artifact_id
    return ids


# ===========================================================================
# File discovery and bundling
# ===========================================================================


def test_discovers_bundle_from_directory() -> None:
    source = RawTraceSource(
        "test", str(FIXTURE_DIR),
        provider=FakeProvider({"events": []}),
    )
    bundles = list(source.iter_bundles())
    assert len(bundles) == 1
    assert len(bundles[0]) == 4


def test_stable_artifact_ids() -> None:
    source1 = RawTraceSource(
        "test", str(FIXTURE_DIR),
        provider=FakeProvider({}),
    )
    source2 = RawTraceSource(
        "test", str(FIXTURE_DIR),
        provider=FakeProvider({}),
    )
    ids1 = [a.artifact_id for b in source1.iter_bundles() for a in b]
    ids2 = [a.artifact_id for b in source2.iter_bundles() for a in b]
    assert ids1 == ids2
    assert len(set(ids1)) == 4


def test_media_type_detection() -> None:
    assert _detect_media_type(Path("test.json")) == "application/json"
    assert _detect_media_type(Path("s.jsonl")) == "application/x-ndjson"
    assert _detect_media_type(Path("n.txt")) == "text/plain"


def test_binary_files_rejected() -> None:
    assert _is_binary(Path("image.png"))
    assert not _is_binary(Path("data.json"))


def test_skips_binary_and_tracks_coverage(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text('{"test": true}')
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    source = RawTraceSource(
        "test", str(tmp_path), provider=FakeProvider({}),
    )
    bundles = list(source.iter_bundles())
    assert len(bundles[0]) == 1
    assert source.coverage.skipped_artifacts[0]["reason"] == "binary file"


def test_bundle_orders_metadata_first() -> None:
    source = RawTraceSource(
        "test", str(FIXTURE_DIR), provider=FakeProvider({}),
    )
    for bundle in source.iter_bundles():
        assert "metadata" in bundle[0].filename


# ===========================================================================
# Size limits
# ===========================================================================


def test_enforces_per_artifact_limit(tmp_path: Path) -> None:
    (tmp_path / "big.json").write_text("x" * 1000)
    source = RawTraceSource(
        "test", str(tmp_path),
        max_artifact_bytes=100, max_bundle_bytes=10_000,
        max_total_bytes=100_000, provider=FakeProvider({}),
    )
    bundles = list(source.iter_bundles())
    assert len(bundles) == 0
    assert "exceeds artifact limit" in (
        source.coverage.skipped_artifacts[0]["reason"]
    )


def test_enforces_total_limit(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"f{i}.json").write_text('{"d":"' + "x" * 100 + '"}')
    source = RawTraceSource(
        "test", str(tmp_path),
        max_artifact_bytes=10_000, max_bundle_bytes=10_000,
        max_total_bytes=500, provider=FakeProvider({}),
    )
    with pytest.raises(ValueError, match="total byte limit exceeded"):
        list(source.iter_bundles())


# ===========================================================================
# Credential redaction
# ===========================================================================


def test_redact_array_style_headers() -> None:
    content = json.dumps({
        "headers": [
            ["Authorization", "Bearer sk-ant-api03-FAKEtoken123"],
            ["x-api-key", "sk-ant-api03-FAKEtoken123"],
            ["Cookie", "session=secret_value_12345"],
            ["Content-Type", "application/json"],
        ]
    })
    redacted = _redact_artifact_content(
        content, "application/json", 10_000
    )
    assert "FAKEtoken" not in redacted
    assert "secret_value" not in redacted
    assert "[REDACTED]" in redacted
    assert "application/json" in redacted


def test_fixture_auth_tokens_redacted() -> None:
    request_content = (FIXTURE_DIR / "request.json").read_text()
    redacted = _redact_artifact_content(
        request_content, "application/json", 100_000
    )
    assert "FAKE_TOKEN_DO_NOT_USE" not in redacted
    assert "sk-ant-api03" not in redacted
    assert "[REDACTED]" in redacted


def test_absolute_paths_never_in_payload() -> None:
    source = RawTraceSource(
        "test", str(FIXTURE_DIR), provider=FakeProvider({}),
    )
    for bundle in source.iter_bundles():
        payload = source._build_payload(bundle)
        assert "/root/" not in payload
        assert "/home/" not in payload
        assert "file://" not in payload


# ===========================================================================
# NormalizedTraceBundle schema strict-mode compliance
# ===========================================================================


def _check_strict(
    schema: dict[str, Any],
    path: str = "root",
    defs: dict[str, Any] | None = None,
) -> list[str]:
    """Recursively check OpenAI strict-mode compliance."""
    if defs is None:
        defs = schema.get("$defs", {})
    errors: list[str] = []
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        errors.extend(_check_strict(defs.get(ref, {}), f"{path}.$ref({ref})", defs))
        return errors
    if schema.get("type") != "object":
        if schema.get("type") == "array" and "items" in schema:
            errors.extend(_check_strict(schema["items"], f"{path}.items", defs))
        if "anyOf" in schema:
            for i, b in enumerate(schema["anyOf"]):
                errors.extend(_check_strict(b, f"{path}.anyOf[{i}]", defs))
        return errors
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    if schema.get("additionalProperties") is not False:
        errors.append(f"{path}: additionalProperties not false")
    missing = set(props.keys()) - required
    if missing:
        errors.append(f"{path}: not in required: {sorted(missing)}")
    for name, prop in props.items():
        if "default" in prop:
            errors.append(f"{path}.{name}: has default")
        errors.extend(_check_strict(prop, f"{path}.{name}", defs))
    return errors


def test_normalized_trace_bundle_schema_strict_compatible() -> None:
    schema = NormalizedTraceBundle.model_json_schema()
    errors = _check_strict(schema)
    assert errors == [], (
        "NormalizedTraceBundle schema not strict-compatible:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


# ===========================================================================
# Evidence validation (deterministic)
# ===========================================================================


def test_validate_bundle_rejects_unknown_artifact_ids() -> None:
    valid_ids = {"art_1", "art_2"}
    bundle = NormalizedTraceBundle.model_validate({
        "identity": {
            "session_id": None, "request_id": None, "parent_id": None,
            "evidence": [_evidence("art_UNKNOWN")],
        },
        "timing": {
            "start_timestamp": None, "end_timestamp": None,
            "latency_ms": None, "evidence": [],
        },
        "http": {
            "status_code": None, "stop_reason": None,
            "model": None, "evidence": [],
        },
        "usage": {
            "input_tokens": None, "output_tokens": None,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None, "evidence": [],
        },
        "messages": [], "tool_calls": [], "tool_results": [],
        "pending_tool_calls": [],
        "coverage": {
            "artifacts_used": [], "artifacts_skipped": [],
            "unresolved_fields": [],
        },
    })
    errors = validate_bundle(bundle, valid_ids)
    assert any("unknown artifact_id" in e for e in errors)


def test_validate_bundle_rejects_duplicate_tool_call_ids() -> None:
    valid_ids = {"art_1"}
    bundle = NormalizedTraceBundle.model_validate({
        "identity": {
            "session_id": None, "request_id": None, "parent_id": None,
            "evidence": [],
        },
        "timing": {
            "start_timestamp": None, "end_timestamp": None,
            "latency_ms": None, "evidence": [],
        },
        "http": {
            "status_code": None, "stop_reason": None,
            "model": None, "evidence": [],
        },
        "usage": {
            "input_tokens": None, "output_tokens": None,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None, "evidence": [],
        },
        "messages": [],
        "tool_calls": [
            {
                "tool_call_id": "t1", "tool_name": "Bash",
                "arguments_json": None, "mcp_server": None,
                "plugin_name": None, "attribution_skill": None,
                "evidence": [_evidence("art_1")],
            },
            {
                "tool_call_id": "t1", "tool_name": "Bash",
                "arguments_json": None, "mcp_server": None,
                "plugin_name": None, "attribution_skill": None,
                "evidence": [_evidence("art_1")],
            },
        ],
        "tool_results": [], "pending_tool_calls": [],
        "coverage": {
            "artifacts_used": ["art_1"], "artifacts_skipped": [],
            "unresolved_fields": [],
        },
    })
    errors = validate_bundle(bundle, valid_ids)
    assert any("duplicate tool_call_id" in e for e in errors)


# ===========================================================================
# Full pipeline (FakeProvider)
# ===========================================================================


def test_produces_canonical_events() -> None:
    artifact_ids = _get_artifact_ids()
    llm_response = _make_bundle_response(artifact_ids)
    fake_provider = FakeProvider(llm_response)
    source = RawTraceSource(
        "relay", str(FIXTURE_DIR), provider=fake_provider,
    )

    events = list(source.iter_events())

    # 2 messages + 2 tool_calls + 2 tool_results = 6 events
    assert len(events) == 6

    tool_uses = [e for e in events if e.event_type == "tool_use"]
    tool_results = [e for e in events if e.event_type == "tool_result"]
    messages = [e for e in events if e.event_type == "message"]
    assert len(tool_uses) == 2
    assert len(tool_results) == 2
    assert len(messages) == 2

    # MCP server attribution from LLM
    assert all(e.mcp_server == "datahub" for e in tool_uses)

    # Tool pairing via finalize_candidates
    for use in tool_uses:
        assert use.paired_event_id is not None
    for result in tool_results:
        assert result.paired_event_id is not None

    # Deterministic enrichment: facts from LLM's bundle, not Python parsing
    for event in events:
        assert event.model == "claude-sonnet-4-20250514"

    # Usage deduplication: only first assistant event gets tokens
    usage_events = [
        e for e in events
        if e.input_tokens is not None and e.input_tokens > 0
    ]
    assert len(usage_events) == 1
    assert usage_events[0].input_tokens == 14042

    # Coverage
    assert source.coverage.total_artifacts == 4
    assert source.coverage.normalized_artifacts == 4
    assert source.coverage.total_events == 6

    # Auth tokens not in LLM payload
    call = fake_provider.calls[0]
    payload = call["payload"]
    assert "FAKE_TOKEN_DO_NOT_USE" not in payload
    assert "sk-ant-api03" not in payload

    # Schema sent is NormalizedTraceBundle, not LlmNormalizationBatch
    schema = call["schema"]
    schema_str = json.dumps(schema)
    assert "NormalizedTraceBundle" in schema_str
    assert "LlmNormalizationCandidate" not in schema_str


# ===========================================================================
# Bounded validation loop
# ===========================================================================


def test_validation_loop_repairs_bad_response() -> None:
    """First response has bad artifact_id, repair fixes it."""
    artifact_ids = _get_artifact_ids()

    good_response = _make_bundle_response(artifact_ids)
    bad_response = _make_bundle_response(artifact_ids)
    bad_response["identity"]["evidence"] = [
        _evidence("NONEXISTENT_ID")
    ]

    class RepairProvider:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self._count = 0

        def request_structured(self, **kwargs: Any) -> ProviderResponse:
            self.calls.append(kwargs)
            self._count += 1
            if self._count == 1:
                return ProviderResponse(
                    raw_text=json.dumps(bad_response)
                )
            return ProviderResponse(
                raw_text=json.dumps(good_response)
            )

    provider = RepairProvider()
    source = RawTraceSource(
        "relay", str(FIXTURE_DIR), provider=provider,
    )

    events = list(source.iter_events())
    assert len(events) == 6
    assert len(provider.calls) == 2
    assert "repair" in provider.calls[1]["operation"]


def test_validation_loop_fails_after_max_attempts() -> None:
    """After max repairs, raises with the error details."""
    artifact_ids = _get_artifact_ids()
    bad_response = _make_bundle_response(artifact_ids)
    bad_response["identity"]["evidence"] = [
        _evidence("ALWAYS_WRONG")
    ]
    fake = FakeProvider(bad_response)
    source = RawTraceSource(
        "relay", str(FIXTURE_DIR), provider=fake,
    )

    with pytest.raises(RuntimeError, match="repair attempts"):
        list(source.iter_events())
    # Initial + 2 repairs = 3 calls
    assert len(fake.calls) == 3


# ===========================================================================
# Config validation
# ===========================================================================


def test_config_requires_claude_sdk_normalizer(tmp_path: Path) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text(
        "version: 1\nsources:\n  - id: r\n    type: raw_trace\n"
        "    path: ./c\n    normalizer: rule_based\n"
    )
    with pytest.raises(
        ValidationError, match="require normalizer: claude_sdk"
    ):
        load_config(path)


def test_config_requires_egress(tmp_path: Path) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text(
        "version: 1\nsources:\n  - id: r\n    type: raw_trace\n"
        "    path: ./c\n    normalizer: claude_sdk\n"
    )
    with pytest.raises(
        ValidationError, match="external_data_egress_allowed"
    ):
        load_config(path)


def test_config_accepted(tmp_path: Path) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text(
        "version: 1\nsources:\n  - id: r\n    type: raw_trace\n"
        "    path: ./c\n    normalizer: claude_sdk\n"
        "analysis:\n  external_data_egress_allowed: true\n"
    )
    config = load_config(path)
    assert config.sources[0].type == "raw_trace"
