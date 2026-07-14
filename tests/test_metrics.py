from datetime import UTC, datetime

from loop_engine.metrics import compute_metrics
from loop_engine.models import AssetExposure, OutcomeSignal, TaskRun


def _task(task_id: str, task_type: str, model: str, version: str, corrected: bool) -> TaskRun:
    task = TaskRun(
        task_id=task_id,
        session_id=task_id,
        event_ids=[task_id],
        task_type=task_type,
        started_at=datetime(2026, 7, 13, tzinfo=UTC),
        model_ids=[model],
        asset_exposures=[
            AssetExposure(asset_name="skill", version=version, evidence_event_ids=[task_id])
        ],
    )
    if corrected:
        task.outcome_signals = [
            OutcomeSignal(
                signal_id=f"{task_id}:s",
                task_id=task_id,
                kind="human_correction",
                polarity="negative",
                confidence=1,
                evidence_event_ids=[task_id],
            )
        ]
    return task


def test_metrics_include_requested_groups() -> None:
    tasks = [
        _task("t1", "debugging", "sonnet", "v1", True),
        _task("t2", "debugging", "sonnet", "v1", False),
        _task("t3", "summary", "haiku", "v2", False),
    ]

    metrics = compute_metrics(
        tasks,
        [signal for task in tasks for signal in task.outcome_signals],
        group_by=["task_type", "model", "asset_version"],
    )

    debugging = [
        metric
        for metric in metrics
        if metric.name == "correction_rate"
        and metric.group == {"task_type": "debugging", "model": "sonnet", "asset_version": "v1"}
    ]
    assert debugging[0].value == 0.5


def test_unknown_outcome_and_missing_numeric_values_are_excluded() -> None:
    task = TaskRun(
        task_id="unknown",
        session_id="unknown",
        event_ids=["e1"],
        task_type="debugging",
        started_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    metrics = compute_metrics([task], [])
    by_name = {metric.name: metric for metric in metrics}

    assert by_name["objective_success_rate"].value is None
    assert by_name["objective_success_rate"].coverage == 0
    assert by_name["objective_success_rate"].excluded == 1
    assert by_name["cost_per_task"].value is None
    assert by_name["cost_per_task"].coverage == 0
    assert by_name["latency_per_task_ms"].value is None
    assert by_name["latency_per_task_ms"].coverage == 0


def test_observed_zero_cost_and_latency_remain_valid_measurements() -> None:
    task = TaskRun(
        task_id="zero",
        session_id="zero",
        event_ids=["e1"],
        task_type="debugging",
        started_at=datetime(2026, 7, 13, tzinfo=UTC),
        cost_usd=0.0,
        latency_ms=0,
    )

    metrics = compute_metrics([task], [])
    by_name = {metric.name: metric for metric in metrics}

    assert by_name["cost_per_task"].value == 0.0
    assert by_name["cost_per_task"].coverage == 1.0
    assert by_name["latency_per_task_ms"].value == 0.0
    assert by_name["latency_per_task_ms"].coverage == 1.0
