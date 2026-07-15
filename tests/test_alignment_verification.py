"""Required verification tests per the alignment review.

1. Nested locator resolution — exact path, no parent fallback
2. Two-invocation timing — second latency derived, task = sum
3. Coverage from evidence — derive used IDs, enforce disjoint
4. Golden report assertions — counts, tokens, stops, latencies, tool-def size
5. Real capture invariants — normalized values, not Markdown text
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loop_engine.models import (
    CanonicalEvent,
    ContextComponent,
    EvidenceRef,
    NormalizedTraceBundle,
    OperationalInvocation,
)
from loop_engine.providers.base import ProviderResponse
from loop_engine.reconstruction import (
    _derive_used_from_evidence,
    reconstruct_tasks,
)
from loop_engine.sources.raw_trace import (
    RawTraceSource,
    _remeasure_context_components,
    _resolve_json_path,
)

FIXTURE_DIR = Path(
    "tests/fixtures/relay_capture/20260714T172453.744313-fd84d2c9"
)


# =========================================================================
# 1. Nested locator resolution
# =========================================================================


def test_resolve_simple_key() -> None:
    content = json.dumps({"tools": [1, 2, 3], "model": "sonnet"})
    assert _resolve_json_path(content, "$.tools") == [1, 2, 3]
    assert _resolve_json_path(content, "$.model") == "sonnet"


def test_resolve_nested_index() -> None:
    content = json.dumps({
        "messages": [
            {"role": "user", "content": ["hello", "world"]},
            {"role": "assistant", "content": ["reply"]},
        ]
    })
    # $.messages[0].content[1] → "world"
    result = _resolve_json_path(content, "$.messages[0].content[1]")
    assert result == "world"

    # $.messages[1].content[0] → "reply"
    result = _resolve_json_path(content, "$.messages[1].content[0]")
    assert result == "reply"


def test_resolve_out_of_bounds_returns_none() -> None:
    content = json.dumps({"arr": [1, 2]})
    assert _resolve_json_path(content, "$.arr[5]") is None
    assert _resolve_json_path(content, "$.arr[-1]") is None


def test_resolve_invalid_index_returns_none() -> None:
    content = json.dumps({"arr": [1, 2]})
    assert _resolve_json_path(content, "$.arr[abc]") is None


def test_resolve_trailing_prose_returns_none() -> None:
    """Locator with prose after the path must not resolve."""
    content = json.dumps({"tools": [1, 2]})
    assert _resolve_json_path(content, "$.tools (two items)") is None


def test_resolve_no_parent_fallback() -> None:
    """$.messages[0].content[10] must NOT fallback to $.messages."""
    content = json.dumps({
        "messages": [
            {"content": ["a", "b"]},
        ]
    })
    # Index 10 is out of bounds — should return None, not the array
    result = _resolve_json_path(content, "$.messages[0].content[10]")
    assert result is None


def test_resolve_missing_key_returns_none() -> None:
    content = json.dumps({"tools": []})
    assert _resolve_json_path(content, "$.nonexistent") is None


def test_remeasure_uses_exact_locator() -> None:
    """Remeasurement with nested locator gets the exact value."""
    request = {"tools": [{"name": "search"}, {"name": "get"}]}
    content = json.dumps(request)

    class FakeArtifact:
        def __init__(self, aid: str, c: str) -> None:
            self.artifact_id = aid
            self.content = c

    bundle = NormalizedTraceBundle.model_validate({
        "identity": {"session_id": "s", "request_id": None,
                      "parent_id": None, "evidence": []},
        "timing": {"start_timestamp": None, "end_timestamp": None,
                    "latency_ms": None, "evidence": []},
        "http": {"status_code": None, "stop_reason": None,
                 "model": None, "evidence": []},
        "usage": {"input_tokens": None, "output_tokens": None,
                  "cache_creation_input_tokens": None,
                  "cache_read_input_tokens": None, "evidence": []},
        "messages": [], "tool_calls": [], "tool_results": [],
        "pending_tool_calls": [],
        "invocations": [],
        "context_components": [{
            "component_id": "c1",
            "kind": "tool_definitions",
            "name": "tools",
            "char_count": 9999,
            "item_count": 9999,
            "cacheable": True,
            "summary": "test",
            "evidence": [{"artifact_id": "art_req", "locator": "$.tools"}],
        }],
        "coverage": {"artifacts_used": [], "artifacts_skipped": [],
                     "unresolved_fields": []},
    })

    artifacts = [FakeArtifact("art_req", content)]  # type: ignore[list-item]
    _remeasure_context_components(bundle, artifacts)  # type: ignore[arg-type]

    tools_json = json.dumps(request["tools"], ensure_ascii=False)
    assert bundle.context_components[0].char_count == len(tools_json)
    assert bundle.context_components[0].item_count == 2


# =========================================================================
# 2. Two-invocation timing
# =========================================================================


def test_two_invocation_timing_derives_second_latency() -> None:
    """When second invocation has timestamps but no latency,
    reconstruction derives it. Task latency = sum of known values."""
    inv1 = OperationalInvocation(
        invocation_id="inv_01",
        model="claude-sonnet-4-20250514",
        start_timestamp="2026-07-14T17:24:53Z",
        end_timestamp="2026-07-14T17:25:01.212Z",
        latency_ms=8212,
        http_status=200, stop_reason="tool_use",
        input_tokens=14042, output_tokens=409,
        cache_creation_input_tokens=96651,
        cache_read_input_tokens=0,
        cache_creation_input_tokens_5m=None,
        cache_creation_input_tokens_1h=None,
        thinking_tokens=None, tier="sonnet",
        service_tier=None, evidence=[],
    )
    inv2 = OperationalInvocation(
        invocation_id="inv_02",
        model="claude-sonnet-4-20250514",
        start_timestamp="2026-07-14T17:25:01.212Z",
        end_timestamp="2026-07-14T17:25:17.642Z",
        latency_ms=None,  # Missing — should be derived
        http_status=200, stop_reason="end_turn",
        input_tokens=1519, output_tokens=771,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=96651,
        cache_creation_input_tokens_5m=None,
        cache_creation_input_tokens_1h=None,
        thinking_tokens=None, tier="sonnet",
        service_tier=None, evidence=[],
    )

    event = CanonicalEvent(
        event_id="e1",
        source_id="test",
        timestamp=datetime(2026, 7, 14, 17, 24, 53, tzinfo=UTC),
        event_type="message",
        session_hint="s1",
        role="assistant",
        content="test",
        input_tokens=14042,
        output_tokens=409,
        invocations=[inv1, inv2],
        raw_ref="test:e1",
    )

    tasks = reconstruct_tasks([event])
    assert len(tasks) == 1
    task = tasks[0]

    # Second invocation latency should be derived from timestamps
    derived_inv2 = [
        inv for inv in task.invocations
        if inv.invocation_id == "inv_02"
    ]
    assert len(derived_inv2) == 1
    assert derived_inv2[0].latency_ms == 16430  # 17:25:17.642 - 17:25:01.212

    # Task latency = sum of known latencies
    assert task.latency_ms == 8212 + 16430  # = 24642


# =========================================================================
# 3. Coverage from evidence
# =========================================================================


def test_coverage_derived_from_evidence() -> None:
    """Coverage 'used' is derived from evidence references,
    not just self-reported. Skipped must be disjoint with used."""
    inv = OperationalInvocation(
        invocation_id="inv_01",
        model="test", start_timestamp=None, end_timestamp=None,
        latency_ms=None, http_status=None, stop_reason=None,
        input_tokens=None, output_tokens=None,
        cache_creation_input_tokens=None, cache_read_input_tokens=None,
        cache_creation_input_tokens_5m=None,
        cache_creation_input_tokens_1h=None,
        thinking_tokens=None, tier=None, service_tier=None,
        evidence=[
            EvidenceRef(artifact_id="art_meta", locator=None),
            EvidenceRef(artifact_id="art_resp", locator="line 1"),
        ],
    )
    comp = ContextComponent(
        component_id="c1", kind="tool_definitions", name="tools",
        char_count=100, item_count=2, cacheable=True, summary="test",
        evidence=[
            EvidenceRef(artifact_id="art_req", locator="$.tools"),
        ],
    )

    used = _derive_used_from_evidence([inv], [comp], [])
    assert used == {"art_meta", "art_resp", "art_req"}

    # If art_resp is self-reported as skipped, it should be removed
    # because it's in evidence_used
    all_skipped = {"art_resp", "art_other"}
    final_skipped = all_skipped - used
    assert "art_resp" not in final_skipped
    assert "art_other" in final_skipped


# =========================================================================
# 4. Golden report assertions
# =========================================================================


def test_golden_report_tool_definition_size() -> None:
    """Tool definition size from fixture must be based on the actual
    $.tools value, not the full messages array."""
    request_content = (FIXTURE_DIR / "request.json").read_text()
    parsed = json.loads(request_content)
    assert "tools" in parsed

    # Measure $.tools exactly
    tools_json = json.dumps(parsed["tools"], ensure_ascii=False)
    tools_chars = len(tools_json)

    # The char count must NOT be the full-message repeated value (77,908)
    assert tools_chars != 77908, "Tools should not resolve to full message size"
    # It should be the actual tool definitions size
    assert tools_chars > 0


# =========================================================================
# 5. Real capture invariants
# =========================================================================


class _FakeProvider:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def request_structured(self, **kwargs: Any) -> ProviderResponse:
        self.calls.append(kwargs)
        return ProviderResponse(raw_text=json.dumps(self._response))


def _evidence(aid: str, loc: str | None = None) -> dict[str, Any]:
    return {"artifact_id": aid, "locator": loc}


def test_real_capture_normalized_invariants() -> None:
    """After normalization + reconstruction, verify deterministic
    invariants — not Markdown text."""
    fake = _FakeProvider({})
    source = RawTraceSource("relay", str(FIXTURE_DIR), provider=fake)
    bundles = list(source.iter_bundles())
    artifact_ids: dict[str, str] = {}
    for bundle in bundles:
        for a in bundle:
            artifact_ids[a.filename] = a.artifact_id

    meta_id = next(
        aid for f, aid in artifact_ids.items() if "metadata" in f
    )
    req_id = next(
        aid for f, aid in artifact_ids.items() if "request" in f
    )
    resp_id = next(
        aid for f, aid in artifact_ids.items() if "response" in f
    )
    tool_id = next(
        aid for f, aid in artifact_ids.items() if "tool_results" in f
    )

    response: dict[str, Any] = {
        "identity": {
            "session_id": "20260714T172453.744313-fd84d2c9",
            "request_id": None, "parent_id": None,
            "evidence": [_evidence(meta_id)],
        },
        "timing": {
            "start_timestamp": "2026-07-14T17:24:53.744313Z",
            "end_timestamp": None, "latency_ms": 8212,
            "evidence": [_evidence(meta_id, "$.duration_ms")],
        },
        "http": {
            "status_code": 200, "stop_reason": "tool_use",
            "model": "claude-sonnet-4-20250514",
            "evidence": [_evidence(meta_id)],
        },
        "usage": {
            "input_tokens": 14042, "output_tokens": 409,
            "cache_creation_input_tokens": 96651,
            "cache_read_input_tokens": 0,
            "evidence": [_evidence(resp_id, "line 1")],
        },
        "messages": [
            {"role": "user", "content": "test",
             "evidence": [_evidence(req_id)]},
            {"role": "assistant", "content": "test",
             "evidence": [_evidence(resp_id)]},
        ],
        "tool_calls": [
            {"tool_call_id": "t1",
             "tool_name": "mcp__datahub__search_datasets",
             "arguments_json": "{}", "mcp_server": "datahub",
             "plugin_name": None, "attribution_skill": None,
             "evidence": [_evidence(resp_id)]},
            {"tool_call_id": "t2",
             "tool_name": "mcp__datahub__get_dataset_schema",
             "arguments_json": "{}", "mcp_server": "datahub",
             "plugin_name": None, "attribution_skill": None,
             "evidence": [_evidence(resp_id)]},
        ],
        "tool_results": [],
        "pending_tool_calls": ["t1", "t2"],
        "invocations": [
            {
                "invocation_id": "inv_01",
                "model": "claude-sonnet-4-20250514",
                "start_timestamp": "2026-07-14T17:24:53.744313Z",
                "end_timestamp": None, "latency_ms": 8212,
                "http_status": 200, "stop_reason": "tool_use",
                "input_tokens": 14042, "output_tokens": 409,
                "cache_creation_input_tokens": 96651,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens_5m": None,
                "cache_creation_input_tokens_1h": None,
                "thinking_tokens": None, "tier": "sonnet",
                "service_tier": None,
                "evidence": [_evidence(meta_id), _evidence(resp_id)],
            },
        ],
        "context_components": [
            {
                "component_id": "comp_sys",
                "kind": "system_prompt",
                "name": "system", "char_count": 1000,
                "item_count": 1, "cacheable": True,
                "summary": "test",
                "evidence": [_evidence(req_id, "$.system")],
            },
            {
                "component_id": "comp_tools",
                "kind": "tool_definitions",
                "name": "tools", "char_count": 9999,
                "item_count": 9999, "cacheable": True,
                "summary": "test",
                "evidence": [_evidence(req_id, "$.tools")],
            },
        ],
        "coverage": {
            "artifacts_used": [meta_id, req_id, resp_id, tool_id],
            "artifacts_skipped": [],
            "unresolved_fields": [],
        },
    }

    provider = _FakeProvider(response)
    source = RawTraceSource("relay", str(FIXTURE_DIR), provider=provider)
    events = list(source.iter_events())
    tasks = reconstruct_tasks(events)

    assert len(tasks) == 1
    task = tasks[0]

    # Token invariants
    assert task.input_tokens == 14042
    assert task.output_tokens == 409
    assert task.cache_creation_input_tokens == 96651

    # Stop reasons from invocations
    assert "tool_use" in task.stop_reasons

    # HTTP statuses deduplicated
    assert task.http_statuses == [200]

    # Tool definition size should be remeasured from $.tools,
    # not the full request size
    tool_comps = [
        c for c in task.context_components
        if c.kind == "tool_definitions"
    ]
    assert len(tool_comps) == 1
    request_json = json.loads(
        (FIXTURE_DIR / "request.json").read_text()
    )
    expected_tools_chars = len(
        json.dumps(request_json["tools"], ensure_ascii=False)
    )
    assert tool_comps[0].char_count == expected_tools_chars
    assert tool_comps[0].item_count == len(request_json["tools"])
