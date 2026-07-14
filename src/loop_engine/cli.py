from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from loop_engine.config import load_config
from loop_engine.engine import LoopEngine
from loop_engine.experiments import evaluate_experiment
from loop_engine.models import TaskRun
from loop_engine.security import secure_write_text

app = typer.Typer(no_args_is_help=True, help="Enterprise AI learning loop engine.")


@app.command()
def validate(config: Annotated[Path, typer.Option("--config", "-c")]) -> None:
    """Validate configuration without reading logs."""
    loaded = load_config(config)
    typer.echo(f"valid: {len(loaded.sources)} source(s), analyzer={loaded.analysis.provider}")


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", "-c")],
    workspace: Annotated[Path | None, typer.Option("--workspace")] = None,
) -> None:
    """Run the complete Phase 0 analysis loop."""
    result = LoopEngine.from_config(config, workspace_override=workspace).run()
    typer.echo(
        json.dumps(
            {
                "event_count": result.event_count,
                "task_count": result.task_count,
                "signal_count": len(result.signals),
                "proposal_count": len(result.proposals),
                "output_directory": str(result.output_directory),
            },
            ensure_ascii=False,
        )
    )


@app.command("experiment-evaluate")
def experiment_evaluate(
    tasks: Annotated[Path, typer.Option("--tasks")],
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    task_type: Annotated[str, typer.Option("--task-type")],
    asset: Annotated[str, typer.Option("--asset")],
    baseline: Annotated[str, typer.Option("--baseline")],
    candidate: Annotated[str, typer.Option("--candidate")],
    metric: Annotated[str, typer.Option("--metric")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Compare one metric across baseline and candidate asset versions."""
    payload = json.loads(tasks.read_text())
    task_runs = [TaskRun.model_validate(item) for item in payload]
    result = evaluate_experiment(
        experiment_id=experiment_id,
        tasks=task_runs,
        task_type=task_type,
        asset_name=asset,
        baseline_version=baseline,
        candidate_version=candidate,
        metric_name=metric,
    )
    serialized = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if output is not None:
        secure_write_text(output, serialized + "\n")
    typer.echo(serialized)


if __name__ == "__main__":
    app()
