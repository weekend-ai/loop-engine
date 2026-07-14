from pathlib import Path

from loop_engine.engine import LoopEngine


def test_config_relative_paths_resolve_from_config_directory(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/claude/session-1.jsonl").resolve()
    config_dir = tmp_path / "config-dir"
    config_dir.mkdir()
    config = config_dir / "loop.yaml"
    config.write_text(
        f"""version: 1
workspace: ./state
sources:
  - id: claude
    type: claude_code_jsonl
    path: {fixture}
analysis:
  provider: rule_based
output:
  directory: ./reports
  formats: [json, markdown]
"""
    )

    engine = LoopEngine.from_config(config)
    result = engine.run()

    assert (config_dir / "state" / "loop.duckdb").exists()
    assert result.output_directory == config_dir / "reports"
    assert (config_dir / "reports" / "report.md").exists()
