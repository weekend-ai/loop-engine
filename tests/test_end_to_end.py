from pathlib import Path
from stat import S_IMODE

from loop_engine.engine import LoopEngine


def test_run_closes_analysis_loop(tmp_path: Path) -> None:
    config_path = Path("tests/fixtures/config.yaml")

    result = LoopEngine.from_config(config_path, workspace_override=tmp_path).run()

    assert result.event_count >= 7
    assert result.task_count == 2
    assert any(signal.kind == "human_correction" for signal in result.signals)
    assert any(metric.name == "correction_rate" for metric in result.metrics)
    assert result.proposals
    assert (tmp_path / "output" / "task_runs.json").exists()
    assert (tmp_path / "output" / "metrics.json").exists()
    assert (tmp_path / "output" / "improvement_proposals.json").exists()
    report = tmp_path / "output" / "report.md"
    report = (result.output_directory / "report.md").read_text()
    assert "# AI Learning Loop Report" in report
    assert "asset_version=" in report
    assert S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert S_IMODE((tmp_path / "output").stat().st_mode) == 0o700
    assert S_IMODE((tmp_path / "loop.duckdb").stat().st_mode) == 0o600
    assert S_IMODE((tmp_path / "output" / "task_runs.json").stat().st_mode) == 0o600
