from __future__ import annotations

import json
from pathlib import Path

from loop_engine.models import CanonicalEvent, OutcomeSignal, TaskRun
from loop_engine.security import secure_directory, secure_file


class DuckDBStore:
    def __init__(self, path: Path) -> None:
        import duckdb

        secure_directory(path.parent)
        self.connection = duckdb.connect(str(path))
        secure_file(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS events (event_id VARCHAR PRIMARY KEY, payload JSON)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS tasks (task_id VARCHAR PRIMARY KEY, payload JSON)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS signals (signal_id VARCHAR PRIMARY KEY, payload JSON)"
        )

    def persist(
        self,
        events: list[CanonicalEvent],
        tasks: list[TaskRun],
        signals: list[OutcomeSignal],
    ) -> None:
        self.connection.execute("BEGIN TRANSACTION")
        try:
            for table in ("events", "tasks", "signals"):
                self.connection.execute(f"DELETE FROM {table}")  # noqa: S608
            for table, key, items in (
                ("events", "event_id", events),
                ("tasks", "task_id", tasks),
                ("signals", "signal_id", signals),
            ):
                for item in items:
                    item_key = getattr(item, key)
                    payload = json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
                    self.connection.execute(
                        f"INSERT INTO {table} VALUES (?, ?)",  # noqa: S608
                        [item_key, payload],
                    )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self.connection.close()
