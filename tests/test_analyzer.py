from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from loop_engine.analyzers.claude_sdk import ClaudeSdkAnalyzer
from loop_engine.models import AssetExposure, CanonicalEvent, TaskRun, TaskSemanticAnalysis
from loop_engine.providers.base import ProviderResponse


class FakeProvider:
    """Test provider that returns a canned response or raises."""

    def __init__(self, response: dict[str, Any] | Exception) -> None:
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
            "system": system_prompt,
            "payload": payload,
            "schema": schema,
            "max_output_tokens": max_output_tokens,
            "operation": operation,
        })
        if isinstance(self._response, Exception):
            raise self._response
        return ProviderResponse(raw_text=json.dumps(self._response))


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
    provider = FakeProvider(_analysis_response())
    task, events = _task_and_events()

    analysis = ClaudeSdkAnalyzer(
        model="litellm-claude", provider=provider, repair=False
    ).analyze(task, events)

    assert analysis.task_type == "coding_debugging"
    assert analysis.signals[0].evidence_event_ids == ["e2"]
    call = provider.calls[0]
    assert call["model"] == "litellm-claude"
    serialized_input = call["payload"]
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
    provider = FakeProvider(_analysis_response())
    task, events = _task_and_events()

    ClaudeSdkAnalyzer(
        provider=provider,
        redact_before_egress=False,
        repair=False,
    ).analyze(task, events)

    serialized_input = provider.calls[0]["payload"]
    assert "super-secret-value" in serialized_input
    assert "MODELSECRET" in serialized_input


def test_claude_sdk_analyzer_surfaces_provider_error() -> None:
    provider = FakeProvider(RuntimeError("credentials unavailable"))
    task, events = _task_and_events()

    with pytest.raises(RuntimeError, match="credentials unavailable"):
        ClaudeSdkAnalyzer(provider=provider, repair=False).analyze(task, events)


def test_claude_sdk_timeout_is_reported() -> None:
    provider = FakeProvider(TimeoutError("timed out"))
    task, events = _task_and_events()

    with pytest.raises(TimeoutError, match="timed out"):
        ClaudeSdkAnalyzer(provider=provider, repair=False).analyze(task, events)


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
    provider = FakeProvider(_analysis_response())
    task, events = _task_and_events()
    events[0].content = "x" * 1000

    with pytest.raises(RuntimeError, match="exceeds configured limit"):
        ClaudeSdkAnalyzer(
            max_input_chars=100, provider=provider, repair=False
        ).analyze(task, events)
    assert provider.calls == []
