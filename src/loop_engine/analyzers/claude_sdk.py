from __future__ import annotations

import json
from typing import Any

from loop_engine.models import CanonicalEvent, TaskRun, TaskSemanticAnalysis
from loop_engine.providers.base import ProviderAdapter
from loop_engine.providers.registry import (
    build_provider,
    request_and_validate,
    resolve_model,
)
from loop_engine.security import redact_value

_SYSTEM_PROMPT = """\
Analyze one enterprise AI task run. Return the requested structured output.

EVALUATE:
- Context overhead: oversized or irrelevant system prompts, duplicated instructions.
- Cache utilization: distinguish uncached input, cache creation, and cache read tokens.
- Token efficiency: input/output ratio, unnecessary verbosity.
- Tool selection: redundant calls, incorrect arguments, better alternatives.
- Plugin/skill gaps: missing instructions, incomplete attribution.
- Incomplete traces: pending tool calls, missing results.
- Model invocations: intermediate tool_use vs terminal end_turn stops.
- Context composition: measured context sources (system_prompt, skill_instructions,
  tool_definitions, session_context, harness, messages). Context components may
  overlap and must not be summed without disjoint evidence.

FOR EACH FINDING:
- Classify its epistemic_status:
  * observed_fact — directly visible in the data.
  * evidence_backed_inference — supported by evidence but requires interpretation.
  * tentative_recommendation — reasonable suggestion with caveats.
  * unsupported_conclusion — stated without sufficient evidence (avoid these).
- Cite specific event IDs and include short evidence quotes.
- Specify the target_layer (context, prompt, plugin, skill, tool, model).
- Name the target_asset when known (e.g., a specific plugin or skill).
- Note limitations: what evidence is missing, what could invalidate this.

RULES:
- Never interpret a pending tool call as a failure.
- Never treat silence as success or abandonment.
- Root causes are hypotheses, never facts.
- Do not calculate aggregate metrics.
- Use only event_id values present in the supplied bundle.
"""


def _build_analysis_bundle(
    task: TaskRun,
    selected_events: list[CanonicalEvent],
    redact: bool,
    max_chars: int,
) -> dict[str, Any]:
    """Build a complete analysis bundle from normalized trace data.

    Consumes only CanonicalEvent/TaskRun — no provider-specific parsing.
    """
    bundle: dict[str, Any] = {
        "task": {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "intent": task.intent,
            "session_id": task.session_id,
            "model_ids": task.model_ids,
            "tool_names": task.tool_names,
            "input_tokens": task.input_tokens,
            "output_tokens": task.output_tokens,
            "cache_creation_input_tokens": (
                task.cache_creation_input_tokens
            ),
            "cache_read_input_tokens": (
                task.cache_read_input_tokens
            ),
            "latency_ms": task.latency_ms,
            "http_statuses": task.http_statuses,
            "stop_reasons": task.stop_reasons,
            "asset_exposures": [
                exp.model_dump(mode="json")
                for exp in task.asset_exposures
            ],
            "invocations": [
                inv.model_dump(mode="json")
                for inv in task.invocations
            ],
            "context_components": [
                comp.model_dump(mode="json")
                for comp in task.context_components
            ],
        },
        "events": [
            {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "role": event.role,
                "content": event.content,
                "model": event.model,
                "tool_name": event.tool_name,
                "tool_arguments_json": event.tool_arguments_json,
                "tool_result": event.tool_result,
                "tool_call_id": event.tool_call_id,
                "status": event.status,
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
                "cache_creation_input_tokens": (
                    event.cache_creation_input_tokens
                ),
                "cache_read_input_tokens": (
                    event.cache_read_input_tokens
                ),
                "latency_ms": event.latency_ms,
                "http_status": event.http_status,
                "stop_reason": event.stop_reason,
                "mcp_server": event.mcp_server,
                "plugin_name": event.plugin_name,
                "attribution_skill": event.attribution_skill,
                "paired_event_id": event.paired_event_id,
            }
            for event in selected_events
        ],
        "context_profile": {
            "total_input_tokens": task.input_tokens,
            "total_output_tokens": task.output_tokens,
            "cache_creation_input_tokens": (
                task.cache_creation_input_tokens
            ),
            "cache_read_input_tokens": (
                task.cache_read_input_tokens
            ),
            "latency_ms": task.latency_ms,
            "http_statuses": task.http_statuses,
            "stop_reasons": task.stop_reasons,
            "model_ids": task.model_ids,
            "tool_count": len(task.tool_names),
            "event_count": len(selected_events),
            "invocations": [
                inv.model_dump(mode="json")
                for inv in task.invocations
            ],
            "context_components": [
                comp.model_dump(mode="json")
                for comp in task.context_components
            ],
            "coverage_artifacts_used": (
                task.coverage_artifacts_used
            ),
            "coverage_unresolved_fields": (
                task.coverage_unresolved_fields
            ),
            "pending_tool_calls": [
                event.tool_call_id
                for event in selected_events
                if event.event_type == "tool_use"
                and event.tool_call_id
                and not event.paired_event_id
            ],
        },
        "evidence_contract": (
            "Use only event_id values present above in evidence_event_ids. "
            "Include short evidence quotes. "
            "Unknown outcomes must remain unknown. "
            "Never claim a pending tool call failed."
        ),
    }
    if redact:
        bundle = redact_value(bundle, max_chars)
    return bundle


class ClaudeSdkAnalyzer:
    def __init__(
        self,
        model: str = "sonnet",
        timeout_seconds: int = 120,
        max_input_chars: int = 100_000,
        max_event_chars: int = 4_000,
        max_output_tokens: int = 16_384,
        redact_before_egress: bool = True,
        provider_name: str = "anthropic",
        repair: bool = True,
        *,
        client: Any | None = None,
        provider: ProviderAdapter | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_input_chars = max_input_chars
        self.max_event_chars = max_event_chars
        self.max_output_tokens = max_output_tokens
        self.redact_before_egress = redact_before_egress
        self.repair = repair
        self._provider = provider or build_provider(
            provider_name,  # type: ignore[arg-type]
            timeout_seconds=timeout_seconds,
            client=client,
        )

    def analyze(
        self, task: TaskRun, events: list[CanonicalEvent]
    ) -> TaskSemanticAnalysis:
        event_ids = set(task.event_ids)
        selected = [e for e in events if e.event_id in event_ids]
        bundle = _build_analysis_bundle(
            task, selected, self.redact_before_egress, self.max_event_chars,
        )
        serialized = json.dumps(bundle, ensure_ascii=False)
        if len(serialized) > self.max_input_chars:
            raise RuntimeError(
                "Analysis bundle exceeds configured limit "
                f"({len(serialized)} > {self.max_input_chars} characters)"
            )
        result = request_and_validate(
            self._provider,
            model=resolve_model(self.model),
            system_prompt=_SYSTEM_PROMPT,
            payload=serialized,
            target_type=TaskSemanticAnalysis,
            max_output_tokens=self.max_output_tokens,
            operation="semantic analysis",
            repair=self.repair,
        )
        assert isinstance(result, TaskSemanticAnalysis)
        allowed = {e.event_id for e in selected}
        # Validate evidence citations in all finding lists
        for finding_list in (
            result.observations,
            result.inefficiencies,
            result.recommendations,
            result.outcome_signals,
        ):
            for finding in finding_list:
                unknown = set(finding.evidence_event_ids) - allowed
                if unknown:
                    raise RuntimeError(
                        f"Analysis cited unknown event IDs: "
                        f"{sorted(unknown)}"
                    )
        # Legacy signal validation
        for signal in result.signals:
            unknown = set(signal.evidence_event_ids) - allowed
            if unknown:
                raise RuntimeError(
                    f"Analysis cited unknown event IDs: "
                    f"{sorted(unknown)}"
                )
        return result
