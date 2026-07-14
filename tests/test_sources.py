from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest

from loop_engine.reconstruction import reconstruct_tasks
from loop_engine.signals import extract_deterministic_signals
from loop_engine.sources.litellm import (
    LiteLLMLocalJsonSource,
    LiteLLMS3JsonSource,
    events_from_litellm_record,
)


def _record(
    request_id: str,
    start: str,
    messages: list[dict[str, str]],
    response: str,
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "start_time": start,
        "end_time": start,
        "model": "claude-sonnet-4-6",
        "virtual_key": "user-a",
        "messages": messages,
        "response": {"role": "assistant", "content": response},
        "status": "success",
    }


def test_litellm_prefix_chain_emits_only_new_messages(tmp_path: Path) -> None:
    first = _record(
        "r1",
        "2026-07-13T10:00:00Z",
        [{"role": "user", "content": "Fix the test"}],
        "I will inspect it.",
    )
    second = _record(
        "r2",
        "2026-07-13T10:01:00Z",
        [
            {"role": "user", "content": "Fix the test"},
            {"role": "assistant", "content": "I will inspect it."},
            {"role": "user", "content": "Use the existing fixture"},
        ],
        "Done.",
    )
    (tmp_path / "1.json").write_text(json.dumps(first))
    (tmp_path / "2.json").write_text(json.dumps(second))

    events = list(LiteLLMLocalJsonSource("litellm", str(tmp_path / "*.json")).iter_events())

    assert [event.content for event in events] == [
        "Fix the test",
        "I will inspect it.",
        "Use the existing fixture",
        "Done.",
    ]
    assert len({event.session_hint for event in events}) == 1


def test_explicit_session_with_delta_messages_does_not_drop_later_turn(
    tmp_path: Path,
) -> None:
    first = _record(
        "r1",
        "2026-07-13T10:00:00Z",
        [{"role": "user", "content": "First turn"}],
        "First response",
    )
    second = _record(
        "r2",
        "2026-07-13T10:01:00Z",
        [{"role": "user", "content": "Second turn"}],
        "Second response",
    )
    first["metadata"] = {"session_id": "explicit-session"}
    second["metadata"] = {"session_id": "explicit-session"}
    (tmp_path / "1.json").write_text(json.dumps(first))
    (tmp_path / "2.json").write_text(json.dumps(second))

    events = list(LiteLLMLocalJsonSource("litellm", str(tmp_path / "*.json")).iter_events())

    assert [event.content for event in events] == [
        "First turn",
        "First response",
        "Second turn",
        "Second response",
    ]
    assert len({event.session_hint for event in events}) == 1


def test_parallel_prefix_branches_share_inferred_session(tmp_path: Path) -> None:
    root = _record(
        "root",
        "2026-07-13T10:00:00Z",
        [{"role": "user", "content": "Root"}],
        "Root response",
    )
    branch_a = _record(
        "branch-a",
        "2026-07-13T10:01:00Z",
        [
            {"role": "user", "content": "Root"},
            {"role": "assistant", "content": "Root response"},
            {"role": "user", "content": "Branch A"},
        ],
        "A response",
    )
    branch_b = _record(
        "branch-b",
        "2026-07-13T10:02:00Z",
        [
            {"role": "user", "content": "Root"},
            {"role": "assistant", "content": "Root response"},
            {"role": "user", "content": "Branch B"},
        ],
        "B response",
    )
    for index, record in enumerate((root, branch_a, branch_b), start=1):
        (tmp_path / f"{index}.json").write_text(json.dumps(record))

    events = list(LiteLLMLocalJsonSource("litellm", str(tmp_path / "*.json")).iter_events())

    assert len({event.session_hint for event in events}) == 1
    assert {event.content for event in events} >= {"Branch A", "Branch B"}


def test_litellm_records_sort_by_instant_not_timestamp_string(tmp_path: Path) -> None:
    earlier = _record(
        "earlier",
        "2026-07-13T10:00:00+09:00",
        [{"role": "user", "content": "Root"}],
        "Root response",
    )
    later = _record(
        "later",
        "2026-07-13T02:00:00Z",
        [
            {"role": "user", "content": "Root"},
            {"role": "assistant", "content": "Root response"},
            {"role": "user", "content": "Follow-up"},
        ],
        "Done",
    )
    (tmp_path / "earlier.json").write_text(json.dumps(earlier))
    (tmp_path / "later.json").write_text(json.dumps(later))

    events = list(LiteLLMLocalJsonSource("litellm", str(tmp_path / "*.json")).iter_events())

    assert len({event.session_hint for event in events}) == 1
    assert [event.content for event in events] == [
        "Root",
        "Root response",
        "Follow-up",
        "Done",
    ]


def test_litellm_naive_timestamps_are_normalized_to_utc(tmp_path: Path) -> None:
    naive = _record(
        "naive",
        "2026-07-13T01:00:00",
        [{"role": "user", "content": "Root"}],
        "Root response",
    )
    aware = _record(
        "aware",
        "2026-07-13T02:00:00Z",
        [
            {"role": "user", "content": "Root"},
            {"role": "assistant", "content": "Root response"},
            {"role": "user", "content": "Follow-up"},
        ],
        "Done",
    )
    (tmp_path / "naive.json").write_text(json.dumps(naive))
    (tmp_path / "aware.json").write_text(json.dumps(aware))

    events = list(LiteLLMLocalJsonSource("litellm", str(tmp_path / "*.json")).iter_events())

    assert events
    assert all(event.timestamp.tzinfo is not None for event in events)
    assert [event.content for event in events] == [
        "Root",
        "Root response",
        "Follow-up",
        "Done",
    ]


def test_litellm_preserves_explicit_zero_cost() -> None:
    record = _record(
        "zero-cost",
        "2026-07-13T10:00:00Z",
        [{"role": "user", "content": "Task"}],
        "Done",
    )
    record["cost"] = 0.0

    events = list(events_from_litellm_record(record, "litellm", "fixture://zero"))

    assert events[-1].cost_usd == 0.0


def test_litellm_missing_request_id_has_stable_fallback() -> None:
    record = _record(
        "temporary",
        "2026-07-13T10:00:00Z",
        [{"role": "user", "content": "Task"}],
        "Done",
    )
    record.pop("request_id")
    first = list(
        events_from_litellm_record(
            json.loads(json.dumps(record)), "litellm", "fixture://same#record=0"
        )
    )
    second = list(
        events_from_litellm_record(
            json.loads(json.dumps(record)), "litellm", "fixture://same#record=0"
        )
    )

    assert [event.event_id for event in first] == [event.event_id for event in second]
    assert {event.session_hint for event in first} == {event.session_hint for event in second}


def test_litellm_failure_status_with_or_without_response_is_observable(
    tmp_path: Path,
) -> None:
    with_response = _record(
        "failed-with-response",
        "2026-07-13T10:00:00Z",
        [{"role": "user", "content": "Task one"}],
        "Upstream rejected request",
    )
    with_response["status"] = "failure"
    without_response = _record(
        "failed-without-response",
        "2026-07-13T10:01:00Z",
        [{"role": "user", "content": "Task two"}],
        "unused",
    )
    without_response.pop("response")
    without_response["status"] = "failed"
    without_response["error"] = "upstream timeout"
    (tmp_path / "1.json").write_text(json.dumps(with_response))
    (tmp_path / "2.json").write_text(json.dumps(without_response))

    events = list(LiteLLMLocalJsonSource("litellm", str(tmp_path / "*.json")).iter_events())
    tasks = reconstruct_tasks(events)
    signals = extract_deterministic_signals(tasks, events)

    assert sum(event.status == "error" for event in events) == 2
    assert sum(signal.kind == "api_failure" for signal in signals) == 2


class _FakePaginator:
    def paginate(self, **_: object) -> list[dict[str, object]]:
        return [{"Contents": [{"Key": "logs/request.json"}]}]


class _FakeS3Client:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_objects_v2"
        return _FakePaginator()

    def get_object(self, Bucket: str, Key: str) -> dict[str, BytesIO]:  # noqa: N803
        assert Bucket == "bucket"
        assert Key == "logs/request.json"
        return {"Body": BytesIO(self.payload)}


def test_s3_source_reads_litellm_json() -> None:
    payload = json.dumps(
        _record(
            "s3-r1",
            "2026-07-13T10:00:00Z",
            [{"role": "user", "content": "Review this change"}],
            "Looks good.",
        )
    ).encode()
    source = LiteLLMS3JsonSource(
        "s3-source",
        "s3://bucket/logs/",
        client=_FakeS3Client(payload),
    )

    events = list(source.iter_events())

    assert [event.content for event in events] == ["Review this change", "Looks good."]
    assert all(event.raw_ref.startswith("s3://bucket/logs/request.json") for event in events)


def test_s3_prefix_is_treated_as_directory_boundary() -> None:
    payload = json.dumps(
        _record(
            "s3-r1",
            "2026-07-13T10:00:00Z",
            [{"role": "user", "content": "Review"}],
            "Done",
        )
    ).encode()

    class CapturingPaginator:
        prefix: str | None = None

        def paginate(self, **kwargs: object) -> list[dict[str, object]]:
            self.prefix = str(kwargs["Prefix"])
            return [
                {
                    "Contents": [
                        {"Key": "logs/request.json", "Size": len(payload)},
                        {"Key": "logs-archive/ignored.json", "Size": len(payload)},
                    ]
                }
            ]

    paginator = CapturingPaginator()

    class BoundaryClient(_FakeS3Client):
        def get_paginator(self, name: str) -> CapturingPaginator:
            assert name == "list_objects_v2"
            return paginator

    source = LiteLLMS3JsonSource(
        "s3-source", "s3://bucket/logs", client=BoundaryClient(payload)
    )

    events = list(source.iter_events())

    assert paginator.prefix == "logs/"
    assert len(events) == 2


def test_s3_source_rejects_oversized_object() -> None:
    payload = b"{}"

    with pytest.raises(ValueError, match="object size limit"):
        list(
            LiteLLMS3JsonSource(
                "s3-source",
                "s3://bucket/logs/",
                client=_FakeS3Client(payload),
                max_object_bytes=1,
            ).iter_events()
        )


def test_s3_body_read_is_bounded_before_size_check() -> None:
    read_amounts: list[int | None] = []

    class GuardedBody:
        def read(self, amount: int | None = None) -> bytes:
            read_amounts.append(amount)
            if amount is None:
                raise AssertionError("S3 body must never be read without a bound")
            return b"x" * amount

    class GuardedClient(_FakeS3Client):
        def get_object(self, Bucket: str, Key: str) -> dict[str, GuardedBody]:  # noqa: N803
            assert Bucket == "bucket"
            assert Key == "logs/request.json"
            return {"Body": GuardedBody()}

    with pytest.raises(ValueError, match="object size limit"):
        list(
            LiteLLMS3JsonSource(
                "s3-source",
                "s3://bucket/logs/",
                client=GuardedClient(b""),
                max_object_bytes=5,
                max_total_bytes=10,
            ).iter_events()
        )

    assert read_amounts == [6]


def test_litellm_json_records_must_be_objects(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('["not-an-object"]')

    with pytest.raises(ValueError, match="JSON object"):
        list(LiteLLMLocalJsonSource("litellm", str(path)).iter_events())
