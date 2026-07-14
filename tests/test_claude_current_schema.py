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
