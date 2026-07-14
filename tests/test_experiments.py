from datetime import UTC, datetime

import pytest

from loop_engine.experiments import evaluate_experiment
from loop_engine.models import AssetExposure, OutcomeSignal, TaskRun


def _task(task_id: str, version: str, corrected: bool) -> TaskRun:
    task = TaskRun(
        task_id=task_id,
        session_id=task_id,
        event_ids=[f"{task_id}:event"],
        task_type="coding_debugging",
        started_at=datetime(2026, 7, 13, tzinfo=UTC),
        asset_exposures=[
            AssetExposure(
                asset_name="debugging-skill",
                version=version,
                evidence_event_ids=[f"{task_id}:event"],
            )
        ],
    )
    if corrected:
        task.outcome_signals = [
            OutcomeSignal(
                signal_id=f"{task_id}:correction",
                task_id=task_id,
                kind="human_correction",
                polarity="negative",
                confidence=1,
                evidence_event_ids=[f"{task_id}:event"],
            )
        ]
    return task


def test_experiment_compares_asset_versions_deterministically() -> None:
    tasks = [
        _task("baseline-1", "v1", True),
        _task("baseline-2", "v1", True),
        _task("candidate-1", "v2", False),
        _task("candidate-2", "v2", True),
    ]

    result = evaluate_experiment(
        experiment_id="exp-1",
        tasks=tasks,
        task_type="coding_debugging",
        asset_name="debugging-skill",
        baseline_version="v1",
        candidate_version="v2",
        metric_name="correction_rate",
    )

    assert result.baseline_value == 1.0
    assert result.candidate_value == 0.5
    assert result.absolute_delta == -0.5
    assert result.relative_delta == -0.5
    assert result.verdict == "improved"


def test_experiment_rejects_task_exposed_to_both_versions() -> None:
    task = _task("overlap", "v1", True)
    task.asset_exposures.append(
        AssetExposure(
            asset_name="debugging-skill",
            version="v2",
            evidence_event_ids=["overlap:event"],
        )
    )

    with pytest.raises(ValueError, match="both baseline and candidate"):
        evaluate_experiment(
            experiment_id="exp-overlap",
            tasks=[task],
            task_type="coding_debugging",
            asset_name="debugging-skill",
            baseline_version="v1",
            candidate_version="v2",
            metric_name="correction_rate",
        )


def test_objective_experiment_excludes_unknown_outcomes() -> None:
    baseline_success = _task("baseline-success", "v1", False)
    baseline_success.outcome_signals.append(
        OutcomeSignal(
            signal_id="baseline-success:signal",
            task_id=baseline_success.task_id,
            kind="objective_success",
            polarity="positive",
            confidence=1,
            evidence_event_ids=["baseline-success:event"],
        )
    )
    baseline_unknown = _task("baseline-unknown", "v1", False)
    candidate_failure = _task("candidate-failure", "v2", False)
    candidate_failure.outcome_signals.append(
        OutcomeSignal(
            signal_id="candidate-failure:signal",
            task_id=candidate_failure.task_id,
            kind="api_failure",
            polarity="negative",
            confidence=1,
            evidence_event_ids=["candidate-failure:event"],
        )
    )
    candidate_unknown = _task("candidate-unknown", "v2", False)

    result = evaluate_experiment(
        experiment_id="exp-objective",
        tasks=[
            baseline_success,
            baseline_unknown,
            candidate_failure,
            candidate_unknown,
        ],
        task_type="coding_debugging",
        asset_name="debugging-skill",
        baseline_version="v1",
        candidate_version="v2",
        metric_name="objective_success_rate",
    )

    assert result.baseline_n == 1
    assert result.candidate_n == 1
    assert result.baseline_value == 1.0
    assert result.candidate_value == 0.0
