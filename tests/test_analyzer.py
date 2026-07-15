from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from loop_engine.analyzers.claude_sdk import ClaudeSdkAnalyzer
from loop_engine.models import AssetExposure, CanonicalEvent, TaskRun, TaskSemanticAnalysis


class FakeMessages:
    def __init__(self, response: dict[str, Any] | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self.response))]
        )


class FakeClient:
    def __init__(self, response: dict[str, Any] | Exception) -> None:
        self.messages = FakeMessages(response)


def _task_and_events() -> tuple[TaskRun, list[CanonicalEvent]]:
    task = TaskRun(
        task_id="t1",
        session_id="s1",
        event_ids=["e1", "e2"],
        intent="Fix the test",
        started_at=datetime(2026, 7, 13, tzinfo=UTC),
        model_ids=["api_key=MODELSECRET"],
        tool_names=["api_key=TOOLSECRET"],
        asset_exposures=[
            AssetExposure(
                asset_name="secret=ASSETSECRET",
                version="token=VERSIONSECRET",
                evidence_event_ids=["e1"],
            )
        ],
    )
    events = [
        CanonicalEvent(
            event_id="e1",
            source_id="fixture",
            timestamp=task.started_at,
            event_type="message",
            session_hint="s1",
            role="user",
            content="Fix the authentication test",
            raw_ref="fixture:e1",
        ),
        CanonicalEvent(
            event_id="e2",
            source_id="fixture",
            timestamp=datetime(2026, 7, 13, 0, 1, tzinfo=UTC),
            event_type="message",
            session_hint="s1",
            role="user",
            content=(
                'Do not change the schema; api_key=super-secret-value; '
                '{"password":"JSONSECRET"}'
            ),
            raw_ref="fixture:e2",
        ),
    ]
    return task, events


def _analysis_response() -> dict[str, Any]:
    return {
        "task_type": "coding_debugging",
        "intent": "Fix authentication test",
        "signals": [
            {
                "kind": "human_correction",
                "subtype": "constraint_reminder",
                "polarity": "negative",
                "confidence": 0.93,
                "evidence_event_ids": ["e2"],
                "evidence_quotes": ["Do not change the schema"],
            }
        ],
        "root_cause_hypotheses": ["The active skill ignored a user constraint."],
    }


def test_claude_sdk_analyzer_uses_schema_and_preserves_evidence() -> None:
    client = FakeClient(_analysis_response())
    task, events = _task_and_events()

    analysis = ClaudeSdkAnalyzer(model="litellm-claude", client=client).analyze(task, events)

    assert analysis.task_type == "coding_debugging"
    assert analysis.signals[0].evidence_event_ids == ["e2"]
    call = client.messages.calls[0]
    assert call["model"] == "litellm-claude"
    assert call["output_config"]["format"]["type"] == "json_schema"
    serialized_input = call["messages"][0]["content"]
    assert "evidence_event_ids" in serialized_input
    for secret in (
        "super-secret-value",
        "MODELSECRET",
        "TOOLSECRET",
        "ASSETSECRET",
        "VERSIONSECRET",
        "JSONSECRET",
    ):
        assert secret not in serialized_input
    assert "[REDACTED]" in serialized_input
    assert "tool_result" not in serialized_input


def test_claude_sdk_analyzer_can_disable_redaction() -> None:
    client = FakeClient(_analysis_response())
    task, events = _task_and_events()

    ClaudeSdkAnalyzer(
        client=client,
        redact_before_egress=False,
    ).analyze(task, events)

    serialized_input = client.messages.calls[0]["messages"][0]["content"]
    assert "super-secret-value" in serialized_input
    assert "MODELSECRET" in serialized_input


def test_claude_sdk_analyzer_surfaces_sdk_error() -> None:
    client = FakeClient(RuntimeError("credentials unavailable"))
    task, events = _task_and_events()

    with pytest.raises(RuntimeError, match="credentials unavailable"):
        ClaudeSdkAnalyzer(client=client).analyze(task, events)


def test_claude_sdk_timeout_is_reported() -> None:
    client = FakeClient(TimeoutError("timed out"))
    task, events = _task_and_events()

    with pytest.raises(RuntimeError, match="timed out"):
        ClaudeSdkAnalyzer(client=client).analyze(task, events)


def test_semantic_signal_requires_evidence_event_id() -> None:
    with pytest.raises(ValidationError):
        TaskSemanticAnalysis.model_validate(
            {
                "task_type": "coding_debugging",
                "intent": "Fix test",
                "signals": [
                    {
                        "kind": "human_correction",
                        "polarity": "negative",
                        "confidence": 0.9,
                        "evidence_event_ids": [],
                        "evidence_quotes": [],
                    }
                ],
            }
        )


def test_claude_sdk_rejects_oversized_bundle_before_request() -> None:
    client = FakeClient(_analysis_response())
    task, events = _task_and_events()
    events[0].content = "x" * 1000

    with pytest.raises(RuntimeError, match="exceeds configured limit"):
        ClaudeSdkAnalyzer(max_input_chars=100, client=client).analyze(task, events)
    assert client.messages.calls == []
