from __future__ import annotations

import hashlib
import re
from typing import Literal

from loop_engine.models import CanonicalEvent, OutcomeSignal, TaskRun

_CORRECTION_PATTERNS = (
    re.compile(r"\b(no|wrong|incorrect|don't|do not|instead|actually)\b", re.IGNORECASE),
    re.compile(r"\b(不对|不要|错了|应该|改成|而不是)\b"),
)
_SUCCESS_PATTERNS = (
    re.compile(r"\b[1-9]\d* passed\b", re.IGNORECASE),
    re.compile(r"\b(success|succeeded|all tests pass)\b", re.IGNORECASE),
)


def stable_signal_id(
    task_id: str, kind: str, subtype: str | None, evidence_event_ids: list[str]
) -> str:
    signal_key = "\0".join((task_id, kind, subtype or "", *sorted(evidence_event_ids)))
    return f"sig:{hashlib.sha256(signal_key.encode()).hexdigest()[:16]}"


def _new_signal(
    task: TaskRun,
    kind: str,
    subtype: str,
    polarity: Literal["positive", "negative", "neutral", "unknown"],
    event: CanonicalEvent,
    quote: str,
    confidence: float,
) -> OutcomeSignal:
    return OutcomeSignal(
        signal_id=stable_signal_id(task.task_id, kind, subtype, [event.event_id]),
        task_id=task.task_id,
        kind=kind,
        subtype=subtype,
        polarity=polarity,
        confidence=confidence,
        evidence_event_ids=[event.event_id],
        evidence_quotes=[quote[:500]],
    )


def extract_deterministic_signals(
    tasks: list[TaskRun], events: list[CanonicalEvent]
) -> list[OutcomeSignal]:
    by_id = {event.event_id: event for event in events}
    all_signals: list[OutcomeSignal] = []
    for task in tasks:
        task_events = [by_id[event_id] for event_id in task.event_ids]
        seen_first_user = False
        for event in task_events:
            text = event.content or event.tool_result or ""
            if event.role == "user":
                if not seen_first_user:
                    seen_first_user = True
                elif any(pattern.search(text) for pattern in _CORRECTION_PATTERNS):
                    all_signals.append(
                        _new_signal(
                            task,
                            "human_correction",
                            "constraint_or_factual_correction",
                            "negative",
                            event,
                            text,
                            0.85,
                        )
                    )
            if event.status == "error":
                is_tool_failure = bool(event.tool_name) or event.event_type in {
                    "tool_use",
                    "tool_result",
                }
                all_signals.append(
                    _new_signal(
                        task,
                        "tool_failure" if is_tool_failure else "api_failure",
                        event.tool_name or event.event_type,
                        "negative",
                        event,
                        text,
                        1.0,
                    )
                )
            if event.status == "success" and any(
                pattern.search(text) for pattern in _SUCCESS_PATTERNS
            ):
                all_signals.append(
                    _new_signal(
                        task,
                        "objective_success",
                        "test_or_tool_success",
                        "positive",
                        event,
                        text,
                        0.95,
                    )
                )
        task.outcome_signals = [signal for signal in all_signals if signal.task_id == task.task_id]
    return all_signals
