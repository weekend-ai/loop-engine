from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from loop_engine.config import load_config
from loop_engine.metrics import compute_metrics
from loop_engine.models import CanonicalEventCandidate, RawRecordEnvelope
from loop_engine.reconstruction import reconstruct_tasks
from loop_engine.signals import extract_deterministic_signals
from loop_engine.sources.claude_jsonl import ClaudeCodeJsonlSource
from loop_engine.sources.claude_normalization import (
    ClaudeCliRecordNormalizer,
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
    path = Path(
        "tests/fixtures/claude_sessions/"
        "40e79c0c-b67f-436f-b0c1-650f9e6a5357.jsonl"
    )

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


def test_claude_normalizer_redacts_preserves_unknown_fields_and_finalizes_pairing() -> None:
    captured: dict[str, object] = {}

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["input"] = kwargs["input"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "structured_output": {
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
                }
            ),
            stderr="",
        )

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
    normalizer = ClaudeCliRecordNormalizer(runner=runner, timeout_seconds=9)

    candidates = normalizer.normalize(envelopes)
    events = finalize_candidates(envelopes, candidates)

    serialized = str(captured["input"])
    assert "future_field" in serialized
    assert "RAWSECRET" not in serialized
    assert "[REDACTED]" in serialized
    assert "file:///tmp/session" not in serialized
    assert "--bare" in captured["command"]
    assert events[0].paired_event_id == events[1].event_id
    assert events[1].paired_event_id == events[0].event_id
    assert events[0].mcp_server == "github"
    assert events[0].plugin_name == "plugin-x"


def test_claude_normalizer_rejects_unknown_record_ids() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "structured_output": {
                        "events": [
                            {
                                "record_id": "invented",
                                "timestamp": "2026-07-14T08:00:00Z",
                                "event_type": "message",
                            }
                        ]
                    }
                }
            ),
            stderr="",
        )

    envelope = RawRecordEnvelope(
        source_id="claude",
        record_id="known",
        raw_ref="file:///tmp/session#line=1",
        line_number=1,
        raw={},
    )

    with pytest.raises(RuntimeError, match="unknown record ID"):
        ClaudeCliRecordNormalizer(runner=runner).normalize([envelope])


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

    called = False

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    envelope = RawRecordEnvelope(
        source_id="claude",
        record_id="large",
        raw_ref="file:///tmp/large#line=1",
        line_number=1,
        raw={"unknown": "x" * 500},
    )
    with pytest.raises(RuntimeError, match="exceeds configured limit"):
        ClaudeCliRecordNormalizer(
            runner=runner, max_input_chars=100, max_record_chars=1000
        ).normalize([envelope])
    assert called is False


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
        "    normalizer: claude_cli\n"
    )

    with pytest.raises(ValidationError, match="external_data_egress_allowed"):
        load_config(path)

    path.write_text(
        path.read_text()
        + "analysis:\n"
        + "  external_data_egress_allowed: true\n"
    )
    assert load_config(path).sources[0].normalizer == "claude_cli"
