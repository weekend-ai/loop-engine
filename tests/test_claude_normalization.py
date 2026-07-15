from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from loop_engine.config import load_config
from loop_engine.metrics import compute_metrics
from loop_engine.models import CanonicalEventCandidate, RawRecordEnvelope
from loop_engine.reconstruction import reconstruct_tasks
from loop_engine.signals import extract_deterministic_signals
from loop_engine.sources.claude_jsonl import ClaudeCodeJsonlSource
from loop_engine.sources.claude_normalization import (
    ClaudeSdkRecordNormalizer,
    finalize_candidates,
)


def test_raw_envelope_accepts_unknown_json_types_and_fallback_skips_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tolerant.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps("string record"),
                json.dumps(["future", {"shape": True}]),
                json.dumps({"timestamp": 123, "message": "not-an-object"}),
                json.dumps({"timestamp": "not-an-iso-time", "message": {}}),
            ]
        )
    )
    source = ClaudeCodeJsonlSource("claude", str(path))

    envelopes = list(source.iter_envelopes())
    events = list(source.iter_events())

    assert [type(envelope.raw) for envelope in envelopes] == [str, list, dict, dict]
    assert events == []


def test_sanitized_session_fallback_handles_tools_pairing_attribution_and_usage() -> None:
    path = Path("tests/fixtures/claude_sessions/synthetic-current-schema.jsonl")

    events = list(ClaudeCodeJsonlSource("claude", str(path)).iter_events())
    tasks = reconstruct_tasks(events)
    signals = extract_deterministic_signals(tasks, events)
    metrics = compute_metrics(tasks, signals)

    assert len(tasks) == 1
    assert tasks[0].tool_names == ["Bash", "mcp__github__search"]
    assert tasks[0].input_tokens == 9953
    assert tasks[0].output_tokens == 591
    assert any(event.mcp_server == "github" for event in events)
    assert any(event.plugin_name == "superpowers" for event in events)
    assert any(event.attribution_skill == "systematic-debugging" for event in events)
    paired = [event for event in events if event.tool_call_id and event.paired_event_id]
    assert {event.tool_call_id for event in paired} == {"toolu_mcp", "toolu_bash"}
    assert {event.status for event in events if event.event_type == "tool_result"} == {
        "success",
        "error",
    }
    assert not any(signal.kind == "human_correction" for signal in signals)
    tool_failure = next(metric for metric in metrics if metric.name == "tool_failure_rate")
    assert tool_failure.value == 1.0


class FakeNormalizerMessages:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self.response))]
        )


class FakeNormalizerClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.messages = FakeNormalizerMessages(response)


def test_claude_normalizer_redacts_preserves_unknown_fields_and_finalizes_pairing() -> None:
    sdk_response = {
        "events": [
            {
                "record_id": "record-1",
                "block_index": 0,
                "timestamp": "2026-07-14T08:00:00Z",
                "event_type": "tool_use",
                "session_hint": "session-1",
                "role": "assistant",
                "tool_name": "mcp__github__search",
                "tool_arguments": {"query": "safe"},
                "tool_call_id": "toolu_1",
                "mcp_server": "github",
                "plugin_name": "plugin-x",
                "attribution_skill": "skill-y",
            },
            {
                "record_id": "record-2",
                "block_index": 0,
                "timestamp": "2026-07-14T08:00:01Z",
                "event_type": "tool_result",
                "session_hint": "session-1",
                "role": "tool",
                "tool_result": "done",
                "content": "done",
                "status": "success",
                "tool_call_id": "toolu_1",
            },
        ]
    }
    client = FakeNormalizerClient(sdk_response)

    envelopes = [
        RawRecordEnvelope(
            source_id="claude",
            record_id="record-1",
            raw_ref="file:///tmp/session#line=1",
            line_number=1,
            raw={
                "future_field": {"nested": [1, 2, 3]},
                "password": "RAWSECRET",
            },
        ),
        RawRecordEnvelope(
            source_id="claude",
            record_id="record-2",
            raw_ref="file:///tmp/session#line=2",
            line_number=2,
            raw={"toolUseResult": "done"},
        ),
    ]
    normalizer = ClaudeSdkRecordNormalizer(client=client)

    candidates = normalizer.normalize(envelopes)
    events = finalize_candidates(envelopes, candidates)

    call = client.messages.calls[0]
    serialized = call["messages"][0]["content"]
    assert "future_field" in serialized
    assert "RAWSECRET" not in serialized
    assert "[REDACTED]" in serialized
    assert "file:///tmp/session" not in serialized
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert events[0].paired_event_id == events[1].event_id
    assert events[1].paired_event_id == events[0].event_id
    assert events[0].mcp_server == "github"
    assert events[0].plugin_name == "plugin-x"


def test_claude_normalizer_rejects_unknown_record_ids() -> None:
    sdk_response = {
        "events": [
            {
                "record_id": "invented",
                "timestamp": "2026-07-14T08:00:00Z",
                "event_type": "message",
            }
        ]
    }
    client = FakeNormalizerClient(sdk_response)

    envelope = RawRecordEnvelope(
        source_id="claude",
        record_id="known",
        raw_ref="file:///tmp/session#line=1",
        line_number=1,
        raw={},
    )

    with pytest.raises(RuntimeError, match="unknown record ID"):
        ClaudeSdkRecordNormalizer(client=client).normalize([envelope])


def test_claude_jsonl_and_normalizer_limits_fail_before_external_call(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large.jsonl"
    path.write_text(json.dumps({"payload": "x" * 200}))
    source = ClaudeCodeJsonlSource(
        "claude", str(path), max_record_bytes=20, max_total_bytes=100
    )
    with pytest.raises(ValueError, match="record size limit"):
        list(source.iter_envelopes())

    sdk_response: dict[str, Any] = {"events": []}
    client = FakeNormalizerClient(sdk_response)

    envelope = RawRecordEnvelope(
        source_id="claude",
        record_id="large",
        raw_ref="file:///tmp/large#line=1",
        line_number=1,
        raw={"unknown": "x" * 500},
    )
    with pytest.raises(RuntimeError, match="exceeds configured limit"):
        ClaudeSdkRecordNormalizer(
            client=client, max_input_chars=100, max_record_chars=1000
        ).normalize([envelope])
    assert client.messages.calls == []


def test_canonical_candidate_enforces_tool_contract() -> None:
    with pytest.raises(ValidationError, match="role='tool'"):
        CanonicalEventCandidate(
            record_id="record",
            timestamp="2026-07-14T08:00:00Z",
            event_type="tool_result",
            role="user",
        )
    with pytest.raises(ValidationError, match="require tool_name"):
        CanonicalEventCandidate(
            record_id="record",
            timestamp="2026-07-14T08:00:00Z",
            event_type="tool_use",
            role="assistant",
        )


def test_claude_source_normalizer_requires_egress_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text(
        "version: 1\n"
        "sources:\n"
        "  - id: claude\n"
        "    type: claude_code_jsonl\n"
        "    path: ./session.jsonl\n"
        "    normalizer: claude_sdk\n"
    )

    with pytest.raises(ValidationError, match="external_data_egress_allowed"):
        load_config(path)

    path.write_text(
        path.read_text()
        + "analysis:\n"
        + "  external_data_egress_allowed: true\n"
    )
    assert load_config(path).sources[0].normalizer == "claude_sdk"
