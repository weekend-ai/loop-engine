import json
from pathlib import Path
from stat import S_IMODE

from typer.testing import CliRunner

from loop_engine.cli import app
from tests.test_experiments import _task

runner = CliRunner()


def test_cli_validates_and_runs_fixture(tmp_path: Path) -> None:
    validated = runner.invoke(app, ["validate", "-c", "tests/fixtures/config.yaml"])
    assert validated.exit_code == 0
    assert "valid" in validated.stdout

    completed = runner.invoke(
        app,
        [
            "run",
            "-c",
            "tests/fixtures/config.yaml",
            "--workspace",
            str(tmp_path),
        ],
    )
    assert completed.exit_code == 0
    payload = json.loads(completed.stdout)
    assert payload["task_count"] == 2


def test_cli_evaluates_experiment(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks = [
        _task("b1", "v1", True),
        _task("b2", "v1", True),
        _task("c1", "v2", False),
        _task("c2", "v2", True),
    ]
    tasks_path.write_text(json.dumps([task.model_dump(mode="json") for task in tasks]))
    output = tmp_path / "experiment.json"

    completed = runner.invoke(
        app,
        [
            "experiment-evaluate",
            "--tasks",
            str(tasks_path),
            "--experiment-id",
            "exp-1",
            "--task-type",
            "coding_debugging",
            "--asset",
            "debugging-skill",
            "--baseline",
            "v1",
            "--candidate",
            "v2",
            "--metric",
            "correction_rate",
            "--output",
            str(output),
        ],
    )

    assert completed.exit_code == 0
    assert json.loads(output.read_text())["verdict"] == "improved"
    assert S_IMODE(output.stat().st_mode) == 0o600
