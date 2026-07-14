from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

from loop_engine.models import MetricResult, OutcomeSignal, TaskRun


def _task_has(task: TaskRun, kind: str) -> bool:
    return any(signal.kind == kind for signal in task.outcome_signals)


def _rate_metric(
    tasks: list[TaskRun],
    name: str,
    predicate: Callable[[TaskRun], bool],
    group: dict[str, str],
) -> MetricResult:
    numerator = sum(1 for task in tasks if predicate(task))
    denominator = len(tasks)
    return MetricResult(
        name=name,
        value=numerator / denominator if denominator else None,
        numerator=numerator,
        denominator=denominator,
        coverage=1.0 if denominator else 0.0,
        confidence="medium" if denominator >= 20 else "low",
        group=group,
    )


def _objective_success_metric(
    tasks: list[TaskRun], group: dict[str, str]
) -> MetricResult:
    known = [
        task
        for task in tasks
        if any(
            _task_has(task, kind)
            for kind in (
                "objective_success",
                "objective_failure",
                "tool_failure",
                "api_failure",
            )
        )
    ]
    numerator = sum(_task_has(task, "objective_success") for task in known)
    denominator = len(known)
    total = len(tasks)
    return MetricResult(
        name="objective_success_rate",
        value=numerator / denominator if denominator else None,
        numerator=numerator,
        denominator=denominator,
        coverage=denominator / total if total else 0,
        confidence="medium" if denominator >= 20 else "low",
        group=group,
        excluded=total - denominator,
    )


def _aggregate(tasks: list[TaskRun], group: dict[str, str]) -> list[MetricResult]:
    count = len(tasks)
    cost_values = [task.cost_usd for task in tasks if task.cost_usd is not None]
    latency_values = [task.latency_ms for task in tasks if task.latency_ms is not None]
    total_turns = sum(len(task.event_ids) for task in tasks)
    return [
        MetricResult(
            name="task_count",
            value=float(count),
            numerator=float(count),
            denominator=1,
            coverage=1.0,
            confidence="high",
            group=group,
        ),
        _rate_metric(
            tasks, "correction_rate", lambda task: _task_has(task, "human_correction"), group
        ),
        _rate_metric(
            tasks, "tool_failure_rate", lambda task: _task_has(task, "tool_failure"), group
        ),
        _rate_metric(
            tasks, "api_failure_rate", lambda task: _task_has(task, "api_failure"), group
        ),
        _objective_success_metric(tasks, group),
        MetricResult(
            name="cost_per_task",
            value=sum(cost_values) / len(cost_values) if cost_values else None,
            numerator=sum(cost_values) if cost_values else None,
            denominator=len(cost_values),
            coverage=len(cost_values) / count if count else 0,
            confidence="medium" if len(cost_values) >= 20 else "low",
            group=group,
            excluded=count - len(cost_values),
        ),
        MetricResult(
            name="latency_per_task_ms",
            value=sum(latency_values) / len(latency_values) if latency_values else None,
            numerator=sum(latency_values) if latency_values else None,
            denominator=len(latency_values),
            coverage=len(latency_values) / count if count else 0,
            confidence="medium" if len(latency_values) >= 20 else "low",
            group=group,
            excluded=count - len(latency_values),
        ),
        MetricResult(
            name="events_per_task",
            value=total_turns / count if count else None,
            numerator=total_turns,
            denominator=count,
            coverage=1.0 if count else 0,
            confidence="high",
            group=group,
        ),
    ]


def _dimension(task: TaskRun, name: str) -> str:
    if name == "task_type":
        return task.task_type
    if name == "model":
        return ",".join(task.model_ids) or "unknown"
    if name == "asset_version":
        versions = sorted({exposure.version for exposure in task.asset_exposures})
        return ",".join(versions) or "none"
    if name == "asset_name":
        names = sorted({exposure.asset_name for exposure in task.asset_exposures})
        return ",".join(names) or "none"
    raise ValueError(f"Unsupported metric grouping dimension: {name}")


def compute_metrics(
    tasks: list[TaskRun],
    signals: list[OutcomeSignal],
    group_by: Sequence[str] | None = None,
) -> list[MetricResult]:
    del signals
    metrics = _aggregate(tasks, {})
    if not group_by:
        return metrics
    grouped: dict[tuple[str, ...], list[TaskRun]] = defaultdict(list)
    for task in tasks:
        grouped[tuple(_dimension(task, name) for name in group_by)].append(task)
    for values, group_tasks in sorted(grouped.items()):
        group = dict(zip(group_by, values, strict=True))
        metrics.extend(_aggregate(group_tasks, group))
    return metrics
