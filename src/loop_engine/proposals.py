from __future__ import annotations

import uuid
from collections import Counter

from loop_engine.models import ImprovementProposal, OutcomeSignal, TaskRun


def build_proposals(
    tasks: list[TaskRun], signals: list[OutcomeSignal]
) -> list[ImprovementProposal]:
    negative = [signal for signal in signals if signal.polarity == "negative"]
    if not negative:
        return []
    task_by_id = {task.task_id: task for task in tasks}
    counts = Counter(task_by_id[signal.task_id].task_type for signal in negative)
    task_type, _ = counts.most_common(1)[0]
    evidence = [
        signal.signal_id for signal in negative if task_by_id[signal.task_id].task_type == task_type
    ]
    kinds = {signal.kind for signal in negative if signal.signal_id in evidence}
    target_layer = "tool" if "tool_failure" in kinds else "instruction"
    return [
        ImprovementProposal(
            proposal_id=f"proposal:{uuid.uuid4().hex[:12]}",
            task_type=task_type,
            title=f"Reduce negative outcome signals for {task_type}",
            hypothesis=(
                "A targeted asset revision can reduce observed corrections and tool failures "
                "without increasing cost or latency beyond configured guardrails."
            ),
            target_layer=target_layer,
            evidence_signal_ids=evidence,
            recommended_experiment=(
                "Create a candidate asset version, hold model and other assets constant, then run "
                "a session-level switchback and compare correction_rate and tool_failure_rate."
            ),
        )
    ]
