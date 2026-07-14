from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from loop_engine.analyzers.claude_cli import ClaudeCliAnalyzer
from loop_engine.models import AssetExposure, CanonicalEvent, TaskRun, TaskSemanticAnalysis


def test_claude_cli_analyzer_uses_schema_and_preserves_evidence() -> None:
    captured: dict[str, object] = {}

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["input"] = kwargs["input"]
        captured["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "structured_output": {
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
                    },
                }
            ),
            stderr="",
        )

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
            timestamp=datetime(2026, 7, 13, tzinfo=UTC),
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
            content="Do not change the schema; api_key=super-secret-value",
            raw_ref="fixture:e2",
        ),
    ]

    analysis = ClaudeCliAnalyzer(
        model="sonnet", timeout_seconds=17, runner=fake_runner
    ).analyze(task, events)

    command = captured["command"]
    assert isinstance(command, list)
    assert analysis.task_type == "coding_debugging"
    assert analysis.signals[0].evidence_event_ids == ["e2"]
    assert "--json-schema" in command
    assert "--bare" in command
    assert "--system-prompt" in command
    assert "--max-turns" not in command
    assert command[command.index("--system-prompt") + 1]
    assert captured["timeout"] == 17
    serialized_input = str(captured["input"])
    assert "evidence_event_ids" in serialized_input
    assert "super-secret-value" not in serialized_input
    assert "MODELSECRET" not in serialized_input
    assert "TOOLSECRET" not in serialized_input
    assert "ASSETSECRET" not in serialized_input
    assert "VERSIONSECRET" not in serialized_input
    assert "[REDACTED]" in serialized_input
    assert "tool_result" not in serialized_input


def test_claude_cli_analyzer_surfaces_json_api_error() -> None:
    def failed_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=json.dumps(
                {
                    "type": "result",
                    "is_error": True,
                    "result": "API Error: credentials unavailable",
                }
            ),
            stderr="",
        )

    task = TaskRun(
        task_id="t1",
        session_id="s1",
        event_ids=["e1"],
        intent="Fix the test",
        started_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    event = CanonicalEvent(
        event_id="e1",
        source_id="fixture",
        timestamp=datetime(2026, 7, 13, tzinfo=UTC),
        event_type="message",
        session_hint="s1",
        role="user",
        content="Fix the test",
        raw_ref="fixture:e1",
    )

    with pytest.raises(RuntimeError, match="credentials unavailable"):
        ClaudeCliAnalyzer(runner=failed_runner).analyze(task, [event])


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


def test_claude_cli_timeout_is_reported() -> None:
    def timeout_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        raise subprocess.TimeoutExpired(command, 5)

    task = TaskRun(
        task_id="t1",
        session_id="s1",
        event_ids=["e1"],
        started_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    event = CanonicalEvent(
        event_id="e1",
        source_id="fixture",
        timestamp=task.started_at,
        event_type="message",
        role="user",
        content="Fix test",
        raw_ref="fixture:e1",
    )

    with pytest.raises(RuntimeError, match="timed out"):
        ClaudeCliAnalyzer(timeout_seconds=5, runner=timeout_runner).analyze(task, [event])


def test_claude_cli_rejects_oversized_bundle_before_subprocess() -> None:
    def unexpected_runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del command, kwargs
        raise AssertionError("runner should not be called")

    task = TaskRun(
        task_id="t1",
        session_id="s1",
        event_ids=["e1"],
        started_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    event = CanonicalEvent(
        event_id="e1",
        source_id="fixture",
        timestamp=task.started_at,
        event_type="message",
        role="user",
        content="x" * 1000,
        raw_ref="fixture:e1",
    )

    with pytest.raises(RuntimeError, match="exceeds configured limit"):
        ClaudeCliAnalyzer(max_input_chars=100, runner=unexpected_runner).analyze(
            task, [event]
        )
