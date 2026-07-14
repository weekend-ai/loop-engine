from datetime import UTC, datetime
from pathlib import Path

import duckdb

from loop_engine.models import CanonicalEvent, OutcomeSignal, TaskRun
from loop_engine.storage import DuckDBStore


def _snapshot(name: str) -> tuple[list[CanonicalEvent], list[TaskRun], list[OutcomeSignal]]:
    event = CanonicalEvent(
        event_id=f"event:{name}",
        source_id="fixture",
        timestamp=datetime(2026, 7, 13, tzinfo=UTC),
        event_type="message",
        session_hint=f"session:{name}",
        role="user",
        content=name,
        raw_ref=f"fixture:{name}",
    )
    task = TaskRun(
        task_id=f"task:{name}",
        session_id=f"session:{name}",
        event_ids=[event.event_id],
        started_at=event.timestamp,
    )
    signal = OutcomeSignal(
        signal_id=f"signal:{name}",
        task_id=task.task_id,
        kind="api_failure",
        polarity="negative",
        confidence=1,
        evidence_event_ids=[event.event_id],
    )
    task.outcome_signals = [signal]
    return [event], [task], [signal]


def test_persist_replaces_current_snapshot_instead_of_accumulating(tmp_path: Path) -> None:
    path = tmp_path / "loop.duckdb"
    store = DuckDBStore(path)
    first = _snapshot("first")
    second = _snapshot("second")
    try:
        store.persist(*first)
        store.persist(*second)
    finally:
        store.close()

    connection = duckdb.connect(str(path), read_only=True)
    try:
        assert connection.execute("select event_id from events").fetchall() == [
            ("event:second",)
        ]
        assert connection.execute("select task_id from tasks").fetchall() == [
            ("task:second",)
        ]
        assert connection.execute("select signal_id from signals").fetchall() == [
            ("signal:second",)
        ]
    finally:
        connection.close()
