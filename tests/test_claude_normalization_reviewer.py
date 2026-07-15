from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from loop_engine.metrics import compute_metrics
from loop_engine.models import RawRecordEnvelope
from loop_engine.providers.base import ProviderResponse
from loop_engine.reconstruction import reconstruct_tasks
from loop_engine.security import redact_text, redact_value
from loop_engine.sources.claude_jsonl import ClaudeCodeJsonlSource
from loop_engine.sources.claude_normalization import (
    ClaudeSdkRecordNormalizer,
    RuleBasedClaudeRecordNormalizer,
    finalize_candidates,
)


class FakeProvider:
    """Test provider that returns a canned response."""

    def __init__(self, response: dict[str, Any] | str | Exception) -> None:
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
            "payload": payload,
            "operation": operation,
        })
        if isinstance(self._response, Exception):
            raise self._response
        if isinstance(self._response, str):
            return ProviderResponse(raw_text=self._response)
        return ProviderResponse(raw_text=json.dumps(self._response))


def test_redact_text_truncation_respects_max_chars() -> None:
    result = redact_text("x" * 500, 100)
    assert result is not None
    assert len(result) == 100
    assert result.endswith("...[TRUNCATED]")


def test_redact_text_rejects_max_chars_smaller_than_marker() -> None:
    with pytest.raises(ValueError, match="truncation marker"):
        redact_text("hello", 5)


def test_redact_value_rejects_pathological_recursion() -> None:
    value: object = "leaf"
    for _ in range(200):
        value = [value]
    with pytest.raises(ValueError, match="recursion depth"):
        redact_value(value, 4000)


def test_redact_text_masks_jwt_gcp_key_and_connection_strings() -> None:
    header = (
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTJ9."
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    text = (
        f"header={header} "
        'gcp={"type":"service_account","private_key":"BEGIN"} '
        "conn=postgresql://user:***@db.example.com:5432/prod "
        "mongo=mongodb+srv://svc:***@cluster.example.net/db"
    )
    redacted = redact_text(text, 4000)
    assert redacted is not None
    assert header not in redacted
    assert "hunter2" not in redacted
    assert "postgresql://" not in redacted
    assert "mongodb+srv://" not in redacted
    assert "BEGIN" not in redacted
    assert "[REDACTED]" in redacted


def test_reconstruction_dedupes_tokens_across_multi_block_message() -> None:
    session = [
        {
            "type": "assistant",
            "uuid": "a1",
            "timestamp": "2026-07-14T08:00:00Z",
            "sessionId": "s-shared",
            "message": {
                "id": "msg_shared",
                "role": "assistant",
                "model": "claude-x",
                "usage": {"input_tokens": 100, "output_tokens": 5},
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"cmd": "true"}},
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": "a1",
            "timestamp": "2026-07-14T08:00:01Z",
            "sessionId": "s-shared",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "is_error": False,
                        "content": "done",
                    }
                ],
            },
        },
    ]
    path = Path("/tmp/token-dedup-fixture.jsonl")
    path.write_text("\n".join(json.dumps(record) for record in session))

    events = list(ClaudeCodeJsonlSource("claude", str(path)).iter_events())
    tasks = reconstruct_tasks(events)
    metrics = compute_metrics(tasks, [])

    usage_events = [event for event in events if event.message_id == "msg_shared"]
    assert len(usage_events) == 2
    assert all(event.input_tokens == 100 for event in usage_events)
    assert tasks[0].input_tokens == 100
    assert tasks[0].output_tokens == 5
    assert next(m.value for m in metrics if m.name == "tool_failure_rate") == 0.0


def test_claude_sdk_normalization_rejects_bad_json() -> None:
    fake_provider = FakeProvider("{not json")

    envelope = RawRecordEnvelope(
        source_id="claude",
        record_id="rec",
        raw_ref="file:///tmp/x#line=1",
        line_number=1,
        raw={"timestamp": "2026-07-14T08:00:00Z", "message": {}},
    )
    with pytest.raises(RuntimeError, match="invalid structured JSON"):
        ClaudeSdkRecordNormalizer(provider=fake_provider, repair=False).normalize([envelope])


def test_rule_based_normalizer_survives_malformed_records() -> None:
    envelopes = [
        RawRecordEnvelope(
            source_id="claude",
            record_id=f"rec-{index}",
            raw_ref=f"file:///tmp/x#line={index}",
            line_number=index,
            raw=raw,
        )
        for index, raw in enumerate(
            [
                None,
                "string only",
                ["list", "only"],
                {"timestamp": "not-a-timestamp"},
                {"timestamp": "2026-07-14T08:00:00Z", "message": "not-a-dict"},
                {"timestamp": "2026-07-14T08:00:00Z", "message": {"content": 123}},
                {
                    "timestamp": "2026-07-14T08:00:00Z",
                    "message": {
                        "content": [
                            {"type": "tool_use", "id": "t", "input": {}},
                            {"type": "tool_result", "is_error": "not-a-bool"},
                            {"type": "future_shape"},
                        ]
                    },
                },
            ],
            start=1,
        )
    ]
    candidates = RuleBasedClaudeRecordNormalizer().normalize(envelopes)
    events = finalize_candidates(envelopes, candidates)
    assert any(event.event_type == "tool_result" for event in events)
    assert not any(
        event.tool_name == "t" and event.event_type == "tool_use" for event in events
    )
