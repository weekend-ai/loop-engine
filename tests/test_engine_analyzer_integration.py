from pathlib import Path

from loop_engine.config import load_config
from loop_engine.engine import LoopEngine
from loop_engine.models import (
    SemanticSignalCandidate,
    TaskSemanticAnalysis,
)


class _FakeSemanticAnalyzer:
    def analyze(self, task: object, events: object) -> TaskSemanticAnalysis:
        del task, events
        return TaskSemanticAnalysis(
            task_type="coding_debugging",
            intent="Fix the authentication test",
            signals=[
                SemanticSignalCandidate(
                    kind="instruction_gap",
                    subtype="missing_constraint",
                    polarity="negative",
                    confidence=0.8,
                    evidence_event_ids=["m4"],
                    evidence_quotes=["don't change the database schema"],
                )
            ],
            root_cause_hypotheses=["The active instruction did not preserve constraints."],
        )


def test_engine_integrates_semantic_analyzer(tmp_path: Path) -> None:
    config = load_config(Path("tests/fixtures/config.yaml"))
    config.sources = [config.sources[0]]
    config.sources[0].path = str(Path("tests/fixtures/claude/*.jsonl").resolve())
    config.analysis.provider = "claude_cli"
    engine = LoopEngine(config, tmp_path, semantic_analyzer=_FakeSemanticAnalyzer())

    result = engine.run()

    assert result.semantic_analyses
    assert any(signal.source == "llm" for signal in result.signals)
    assert (tmp_path / "output" / "semantic_analyses.json").exists()

    repeated = engine.run()
    assert [signal.signal_id for signal in repeated.signals if signal.source == "llm"] == [
        signal.signal_id for signal in result.signals if signal.source == "llm"
    ]
