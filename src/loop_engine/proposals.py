from __future__ import annotations

import hashlib
import uuid
from collections import Counter

from loop_engine.models import (
    ImprovementProposal,
    OutcomeSignal,
    SemanticFinding,
    TaskRun,
    TaskSemanticAnalysis,
)


def _stable_proposal_id(finding: SemanticFinding) -> str:
    """Deterministic ID from category + summary + target."""
    key = f"{finding.category}\0{finding.summary}\0{finding.target_layer}"
    return f"proposal:{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def build_proposals(
    tasks: list[TaskRun],
    signals: list[OutcomeSignal],
    *,
    semantic_analyses: list[TaskSemanticAnalysis] | None = None,
    min_confidence: float = 0.5,
) -> list[ImprovementProposal]:
    """Generate proposals from validated semantic recommendations.

    Sources (in priority order):
    1. SemanticFinding recommendations with evidence + known target layer.
    2. Legacy fallback: negative outcome signals (existing behavior).
    """
    proposals: list[ImprovementProposal] = []
    seen_ids: set[str] = set()

    # --- Source 1: Semantic recommendations ---
    if semantic_analyses:
        task_by_id = {task.task_id: task for task in tasks}
        for _i, analysis in enumerate(semantic_analyses):
            task_type = analysis.task_type
            for rec in analysis.recommendations:
                if rec.confidence < min_confidence:
                    continue
                if not rec.evidence_event_ids:
                    continue
                pid = _stable_proposal_id(rec)
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                proposals.append(ImprovementProposal(
                    proposal_id=pid,
                    task_type=task_type,
                    title=rec.summary,
                    hypothesis=rec.rationale,
                    target_layer=rec.target_layer,
                    evidence_event_ids=rec.evidence_event_ids,
                    recommended_experiment=(
                        rec.expected_benefit or
                        "Evaluate the recommendation in a controlled A/B test."
                    ),
                ))

    # --- Source 2: Legacy negative signal fallback ---
    if not proposals:
        negative = [s for s in signals if s.polarity == "negative"]
        if not negative:
            return []
        task_by_id = {task.task_id: task for task in tasks}
        counts = Counter(
            task_by_id[s.task_id].task_type
            for s in negative
            if s.task_id in task_by_id
        )
        if not counts:
            return []
        task_type, _ = counts.most_common(1)[0]
        evidence = [
            s.signal_id
            for s in negative
            if s.task_id in task_by_id
            and task_by_id[s.task_id].task_type == task_type
        ]
        kinds = {
            s.kind for s in negative if s.signal_id in evidence
        }
        target_layer = "tool" if "tool_failure" in kinds else "instruction"
        proposals.append(ImprovementProposal(
            proposal_id=f"proposal:{uuid.uuid4().hex[:12]}",
            task_type=task_type,
            title=f"Reduce negative outcome signals for {task_type}",
            hypothesis=(
                "A targeted asset revision can reduce observed corrections "
                "and tool failures without increasing cost or latency "
                "beyond configured guardrails."
            ),
            target_layer=target_layer,
            evidence_event_ids=evidence,
            recommended_experiment=(
                "Create a candidate asset version, hold model and other "
                "assets constant, then run a session-level switchback and "
                "compare correction_rate and tool_failure_rate."
            ),
        ))

    return proposals
