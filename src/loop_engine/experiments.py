from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from loop_engine.models import ExperimentResult, TaskRun


def _has_signal(task: TaskRun, kind: str) -> bool:
    return any(signal.kind == kind for signal in task.outcome_signals)


_ALL_TASKS = lambda task: True  # noqa: E731
_OBJECTIVE_OBSERVED = lambda task: any(  # noqa: E731
    _has_signal(task, kind)
    for kind in ("objective_success", "objective_failure", "tool_failure", "api_failure")
)

_METRICS: dict[
    str, tuple[Callable[[TaskRun], bool], bool, Callable[[TaskRun], bool]]
] = {
    "correction_rate": (
        lambda task: _has_signal(task, "human_correction"),
        False,
        _ALL_TASKS,
    ),
    "tool_failure_rate": (
        lambda task: _has_signal(task, "tool_failure"),
        False,
        _ALL_TASKS,
    ),
    "objective_success_rate": (
        lambda task: _has_signal(task, "objective_success"),
        True,
        _OBJECTIVE_OBSERVED,
    ),
}


def _has_asset(task: TaskRun, name: str, version: str) -> bool:
    return any(
        exposure.asset_name == name and exposure.version == version
        for exposure in task.asset_exposures
    )


def evaluate_experiment(
    *,
    experiment_id: str,
    tasks: list[TaskRun],
    task_type: str,
    asset_name: str,
    baseline_version: str,
    candidate_version: str,
    metric_name: str,
) -> ExperimentResult:
    if metric_name not in _METRICS:
        raise ValueError(f"Unsupported experiment metric: {metric_name}")
    predicate, higher_is_better, observable = _METRICS[metric_name]
    eligible = [task for task in tasks if task.task_type == task_type]
    overlap = [
        task
        for task in eligible
        if _has_asset(task, asset_name, baseline_version)
        and _has_asset(task, asset_name, candidate_version)
    ]
    if overlap:
        ids = ", ".join(task.task_id for task in overlap)
        raise ValueError(f"Tasks exposed to both baseline and candidate versions: {ids}")
    baseline = [
        task
        for task in eligible
        if _has_asset(task, asset_name, baseline_version) and observable(task)
    ]
    candidate = [
        task
        for task in eligible
        if _has_asset(task, asset_name, candidate_version) and observable(task)
    ]

    baseline_value = sum(predicate(task) for task in baseline) / len(baseline) if baseline else None
    candidate_value = (
        sum(predicate(task) for task in candidate) / len(candidate) if candidate else None
    )
    verdict: Literal["improved", "regressed", "inconclusive"]
    if baseline_value is None or candidate_value is None:
        absolute_delta = relative_delta = None
        verdict = "inconclusive"
    else:
        absolute_delta = candidate_value - baseline_value
        relative_delta = absolute_delta / baseline_value if baseline_value else None
        if absolute_delta == 0:
            verdict = "inconclusive"
        elif (absolute_delta > 0) == higher_is_better:
            verdict = "improved"
        else:
            verdict = "regressed"

    return ExperimentResult(
        experiment_id=experiment_id,
        metric_name=metric_name,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        absolute_delta=absolute_delta,
        relative_delta=relative_delta,
        baseline_n=len(baseline),
        candidate_n=len(candidate),
        verdict=verdict,
    )
