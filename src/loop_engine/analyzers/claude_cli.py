from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from typing import Any

from loop_engine.models import CanonicalEvent, TaskRun, TaskSemanticAnalysis

Runner = Callable[..., subprocess.CompletedProcess[str]]

_SYSTEM_PROMPT = """Analyze one enterprise AI task run. Return only the requested structured output.
Classify intent and task type, then identify semantic outcome signals. Every signal must cite
specific event IDs from the supplied bundle. Do not treat silence as success or abandonment.
Root causes are hypotheses, never facts. Do not calculate aggregate metrics.
"""

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|token|authorization|password|passwd|secret)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_PROVIDER_TOKEN = re.compile(r"\b(?:sk-ant-|sk-)[A-Za-z0-9_-]{12,}\b")
_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")


def _redact(value: str | None, max_chars: int) -> str | None:
    if value is None:
        return None
    redacted = _SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", value)
    redacted = _BEARER_TOKEN.sub("Bearer [REDACTED]", redacted)
    redacted = _PROVIDER_TOKEN.sub("[REDACTED]", redacted)
    redacted = _AWS_ACCESS_KEY.sub("[REDACTED]", redacted)
    if len(redacted) > max_chars:
        return redacted[:max_chars] + "...[TRUNCATED]"
    return redacted


def _redact_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return _redact(value, max_chars)
    if isinstance(value, list):
        return [_redact_value(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(item, max_chars) for key, item in value.items()}
    return value


class ClaudeCliAnalyzer:
    def __init__(
        self,
        model: str = "sonnet",
        timeout_seconds: int = 120,
        max_input_chars: int = 100_000,
        max_event_chars: int = 4_000,
        *,
        runner: Runner = subprocess.run,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_input_chars = max_input_chars
        self.max_event_chars = max_event_chars
        self.runner = runner

    def analyze(self, task: TaskRun, events: list[CanonicalEvent]) -> TaskSemanticAnalysis:
        event_ids = set(task.event_ids)
        selected = [event for event in events if event.event_id in event_ids]
        bundle: dict[str, Any] = {
            "task": {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "intent": _redact(task.intent, self.max_event_chars),
                "event_ids": task.event_ids,
                "model_ids": task.model_ids,
                "tool_names": task.tool_names,
                "asset_exposures": [
                    exposure.model_dump(mode="json") for exposure in task.asset_exposures
                ],
            },
            "events": [
                {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "role": event.role,
                    "event_type": event.event_type,
                    "content": _redact(event.content, self.max_event_chars),
                    "tool_name": event.tool_name,
                    "status": event.status,
                }
                for event in selected
            ],
            "evidence_contract": (
                "Use only event_id values present above in evidence_event_ids. Include short "
                "evidence quotes. Unknown outcomes must remain unknown."
            ),
        }
        bundle = _redact_value(bundle, self.max_event_chars)
        serialized_bundle = json.dumps(bundle, ensure_ascii=False)
        if len(serialized_bundle) > self.max_input_chars:
            raise RuntimeError(
                "Claude analysis bundle exceeds configured limit "
                f"({len(serialized_bundle)} > {self.max_input_chars} characters)"
            )
        schema = json.dumps(TaskSemanticAnalysis.model_json_schema(), separators=(",", ":"))
        command = [
            "claude",
            "--bare",
            "--print",
            "--system-prompt",
            _SYSTEM_PROMPT,
            "--output-format",
            "json",
            "--json-schema",
            schema,
            "--tools",
            "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--model",
            self.model,
        ]
        try:
            completed = self.runner(
                command,
                input=serialized_bundle,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"Claude CLI analysis timed out after {self.timeout_seconds} seconds"
            ) from error
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = {}
        if completed.returncode != 0 or payload.get("is_error") is True:
            detail = (
                completed.stderr.strip() or str(payload.get("result") or completed.stdout).strip()
            )
            raise RuntimeError(f"Claude CLI analysis failed: {detail}")
        structured = payload.get("structured_output")
        if structured is None:
            result = payload.get("result")
            if isinstance(result, str):
                structured = json.loads(result)
        if structured is None:
            raise RuntimeError("Claude CLI response did not contain structured_output")
        analysis = TaskSemanticAnalysis.model_validate(structured)
        allowed = {event.event_id for event in selected}
        for signal in analysis.signals:
            unknown = set(signal.evidence_event_ids) - allowed
            if unknown:
                raise RuntimeError(f"Claude cited unknown event IDs: {sorted(unknown)}")
        return analysis
