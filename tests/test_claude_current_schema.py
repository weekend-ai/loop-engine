from __future__ import annotations

import json
from pathlib import Path

from loop_engine.metrics import compute_metrics
from loop_engine.reconstruction import reconstruct_tasks
from loop_engine.signals import extract_deterministic_signals
from loop_engine.sources.claude_jsonl import ClaudeCodeJsonlSource


def test_current_claude_schema_skips_metadata_and_extracts_message_blocks(
    tmp_path: Path,
) -> None:
    session_id = "current-schema-session"
    records = [
        {
            "type": "queue-operation",
            "operation": "enqueue",
            "timestamp": "2026-07-13T14:19:51Z",
            "sessionId": session_id,
            "content": "Synthetic prompt",
        },
        {
            "type": "user",
            "uuid": "user-1",
            "parentUuid": None,
            "timestamp": "2026-07-13T14:19:52Z",
            "sessionId": session_id,
            "message": {"role": "user", "content": "Synthetic prompt"},
        },
        {
            "type": "assistant",
            "uuid": "assistant-1",
            "parentUuid": "user-1",
            "timestamp": "2026-07-13T14:20:00Z",
            "sessionId": session_id,
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-test",
                "usage": {"input_tokens": 3, "output_tokens": 4},
                "content": [
                    {
                        "type": "text",
                        "text": "API Error: credentials unavailable",
                    }
                ],
            },
            "error": "unknown",
            "isApiErrorMessage": True,
        },
        {
            "type": "last-prompt",
            "timestamp": "2026-07-13T14:20:01Z",
            "sessionId": session_id,
            "lastPrompt": "Synthetic prompt",
        },
    ]
    path = tmp_path / "session.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))

    events = list(ClaudeCodeJsonlSource("claude", str(path)).iter_events())

    assert [event.role for event in events] == ["user", "assistant"]
    assert events[1].content == "API Error: credentials unavailable"
    assert events[1].model == "claude-sonnet-test"
    assert events[1].input_tokens == 3
    assert events[1].output_tokens == 4
    assert events[1].status == "error"

    tasks = reconstruct_tasks(events)
    signals = extract_deterministic_signals(tasks, events)
    assert any(signal.kind == "api_failure" for signal in signals)
    metrics = compute_metrics(tasks, signals)
    api_failure_rate = next(metric for metric in metrics if metric.name == "api_failure_rate")
    assert api_failure_rate.value == 1.0

    repeated_tasks = reconstruct_tasks(events)
    repeated_signals = extract_deterministic_signals(repeated_tasks, events)
    assert [signal.signal_id for signal in repeated_signals] == [
        signal.signal_id for signal in signals
    ]


def _multi_block_assistant(
    uuid: str,
    parent: str | None,
    message_id: str,
    blocks: list[dict[str, object]],
    usage: dict[str, int],
    ts: str,
    session_id: str,
) -> dict[str, object]:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": ts,
        "sessionId": session_id,
        "message": {
            "id": message_id,
            "role": "assistant",
            "model": "claude-sonnet-test",
            "usage": usage,
            "content": blocks,
        },
    }


def test_current_schema_expands_tool_blocks_and_dedupes_usage_by_message_id(
    tmp_path: Path,
) -> None:
    session_id = "toolblock-session"
    message_id = "msg_vrtx_test"
    usage = {"input_tokens": 9953, "output_tokens": 591}
    records = [
        {
            "type": "user",
            "uuid": "user-1",
            "parentUuid": None,
            "timestamp": "2026-07-13T14:19:52Z",
            "sessionId": session_id,
            "message": {"role": "user", "content": "Run pytest -q"},
        },
        _multi_block_assistant(
            "assistant-1",
            "user-1",
            message_id,
            [
                {"type": "text", "text": "Running pytest."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Bash",
                    "input": {"command": "pytest -q"},
                },
            ],
            usage,
            "2026-07-13T14:20:00Z",
            session_id,
        ),
        {
            "type": "user",
            "uuid": "toolresult-1",
            "parentUuid": "assistant-1",
            "timestamp": "2026-07-13T14:20:05Z",
            "sessionId": session_id,
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "is_error": False,
                        "content": "1 passed",
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "toolresult-2",
            "parentUuid": "assistant-1",
            "timestamp": "2026-07-13T14:20:06Z",
            "sessionId": session_id,
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "is_error": True,
                        "content": "traceback",
                    }
                ],
            },
        },
    ]
    path = tmp_path / "session.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))

    events = list(ClaudeCodeJsonlSource("claude", str(path)).iter_events())

    tool_use_events = [event for event in events if event.event_type == "tool_use"]
    tool_result_events = [event for event in events if event.event_type == "tool_result"]
    assert [event.tool_name for event in tool_use_events] == ["Bash"]
    assert {event.status for event in tool_result_events} == {"success", "error"}
    assert any(event.status == "error" for event in tool_result_events)

    tasks = reconstruct_tasks(events)
    assert len(tasks) == 1
    assert tasks[0].tool_names == ["Bash"]
    assert tasks[0].input_tokens == usage["input_tokens"]
    assert tasks[0].output_tokens == usage["output_tokens"]

    signals = extract_deterministic_signals(tasks, events)
    metrics = compute_metrics(tasks, signals)
    tool_failure_rate = next(metric for metric in metrics if metric.name == "tool_failure_rate")
    assert tool_failure_rate.value == 1.0
