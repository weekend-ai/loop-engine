import json
from pathlib import Path

from loop_engine.reconstruction import reconstruct_tasks
from loop_engine.sources.claude_jsonl import ClaudeCodeJsonlSource


def test_source_id_namespaces_event_and_session_ids(tmp_path: Path) -> None:
    record = {
        "sessionId": "same-session",
        "uuid": "same-event",
        "parentUuid": None,
        "timestamp": "2026-07-13T10:00:00Z",
        "type": "user",
        "message": {"role": "user", "content": "Task"},
    }
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(record))
    second.write_text(json.dumps(record))

    events = [
        *ClaudeCodeJsonlSource("source-a", str(first)).iter_events(),
        *ClaudeCodeJsonlSource("source-b", str(second)).iter_events(),
    ]
    tasks = reconstruct_tasks(events)

    assert len({event.event_id for event in events}) == 2
    assert len({event.session_hint for event in events}) == 2
    assert len(tasks) == 2


def test_claude_fallback_event_id_uses_full_file_identity(tmp_path: Path) -> None:
    record = {
        "sessionId": "same-session",
        "timestamp": "2026-07-13T10:00:00Z",
        "type": "user",
        "message": {"role": "user", "content": "Task"},
    }
    left = tmp_path / "left" / "same.jsonl"
    right = tmp_path / "right" / "same.jsonl"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_text(json.dumps(record))
    right.write_text(json.dumps(record))

    events = [
        *ClaudeCodeJsonlSource("source", str(left)).iter_events(),
        *ClaudeCodeJsonlSource("source", str(right)).iter_events(),
    ]

    assert len({event.event_id for event in events}) == 2


def test_namespace_encoding_is_unambiguous_even_for_direct_adapter_calls(
    tmp_path: Path,
) -> None:
    first_record = {
        "uuid": "c",
        "timestamp": "2026-07-13T10:00:00Z",
        "type": "user",
        "message": {"role": "user", "content": "Task"},
    }
    second_record = {**first_record, "uuid": "b:c"}
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(first_record))
    second.write_text(json.dumps(second_record))

    first_event = next(iter(ClaudeCodeJsonlSource("a:b", str(first)).iter_events()))
    second_event = next(iter(ClaudeCodeJsonlSource("a", str(second)).iter_events()))

    assert first_event.event_id != second_event.event_id
