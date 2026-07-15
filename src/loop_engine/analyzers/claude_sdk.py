from __future__ import annotations

import json
from typing import Any

from loop_engine.anthropic_client import (
    AnthropicClient,
    build_anthropic_client,
    request_structured,
    resolve_model,
)
from loop_engine.models import CanonicalEvent, TaskRun, TaskSemanticAnalysis
from loop_engine.security import redact_value

_SYSTEM_PROMPT = """Analyze one enterprise AI task run. Return only the requested structured output.
Classify intent and task type, then identify semantic outcome signals. Every signal must cite
specific event IDs from the supplied bundle. Do not treat silence as success or abandonment.
Root causes are hypotheses, never facts. Do not calculate aggregate metrics.
"""


class ClaudeSdkAnalyzer:
    def __init__(
        self,
        model: str = "sonnet",
        timeout_seconds: int = 120,
        max_input_chars: int = 100_000,
        max_event_chars: int = 4_000,
        max_output_tokens: int = 4_096,
        redact_before_egress: bool = True,
        *,
        client: AnthropicClient | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_input_chars = max_input_chars
        self.max_event_chars = max_event_chars
        self.max_output_tokens = max_output_tokens
        self.redact_before_egress = redact_before_egress
        self.client = client

    def _client(self) -> AnthropicClient:
        if self.client is None:
            self.client = build_anthropic_client(self.timeout_seconds)
        return self.client

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
        if self.redact_before_egress:
            bundle = redact_value(bundle, self.max_event_chars)
        serialized_bundle = json.dumps(bundle, ensure_ascii=False)
        if len(serialized_bundle) > self.max_input_chars:
            raise RuntimeError(
                "Claude analysis bundle exceeds configured limit "
                f"({len(serialized_bundle)} > {self.max_input_chars} characters)"
            )
        structured = request_structured(
            self._client(),
            model=resolve_model(self.model),
            system_prompt=_SYSTEM_PROMPT,
            payload=serialized_bundle,
            schema=TaskSemanticAnalysis.model_json_schema(),
            max_output_tokens=self.max_output_tokens,
            operation="semantic analysis",
        )
        analysis = TaskSemanticAnalysis.model_validate(structured)
        allowed = {event.event_id for event in selected}
        for signal in analysis.signals:
            unknown = set(signal.evidence_event_ids) - allowed
            if unknown:
                raise RuntimeError(f"Claude cited unknown event IDs: {sorted(unknown)}")
        return analysis
