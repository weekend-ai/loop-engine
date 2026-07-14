from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from typing import Any

from loop_engine.models import CanonicalEvent, TaskRun, TaskSemanticAnalysis
from loop_engine.security import redact_value

Runner = Callable[..., subprocess.CompletedProcess[str]]

_SYSTEM_PROMPT = """Analyze one enterprise AI task run. Return only the requested structured output.
Classify intent and task type, then identify semantic outcome signals. Every signal must cite
specific event IDs from the supplied bundle. Do not treat silence as success or abandonment.
Root causes are hypotheses, never facts. Do not calculate aggregate metrics.
"""

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
                "intent": task.intent,
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
                    "content": event.content,
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
        bundle = redact_value(bundle, self.max_event_chars)
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
