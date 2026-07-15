from __future__ import annotations

from pathlib import Path

from loop_engine.analyzers.base import SemanticAnalyzer
from loop_engine.analyzers.claude_sdk import ClaudeSdkAnalyzer
from loop_engine.config import EngineConfig, load_config
from loop_engine.metrics import compute_metrics
from loop_engine.models import CanonicalEvent, OutcomeSignal, RunResult
from loop_engine.proposals import build_proposals
from loop_engine.reconstruction import reconstruct_tasks
from loop_engine.reporting import write_outputs
from loop_engine.signals import extract_deterministic_signals, stable_signal_id
from loop_engine.sources import build_source
from loop_engine.storage import DuckDBStore


class LoopEngine:
    def __init__(
        self,
        config: EngineConfig,
        workspace: Path,
        semantic_analyzer: SemanticAnalyzer | None = None,
        output_directory: Path | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.output_directory = output_directory or workspace / "output"
        self.semantic_analyzer = semantic_analyzer
        if self.semantic_analyzer is None and config.analysis.provider == "claude_sdk":
            self.semantic_analyzer = ClaudeSdkAnalyzer(
                model=config.analysis.model,
                timeout_seconds=config.analysis.timeout_seconds,
                max_input_chars=config.analysis.max_input_chars,
                max_event_chars=config.analysis.max_event_chars,
                max_output_tokens=config.analysis.max_output_tokens,
                redact_before_egress=config.analysis.redact_before_egress,
                provider_name=config.analysis.provider_name,
                repair=config.analysis.repair,
            )

    @classmethod
    def from_config(cls, path: Path, workspace_override: Path | None = None) -> LoopEngine:
        config_path = path.resolve()
        base_directory = config_path.parent
        config = load_config(config_path)
        for source in config.sources:
            if source.path is not None and not Path(source.path).is_absolute():
                source.path = str(base_directory / source.path)
        if workspace_override is not None:
            workspace = workspace_override
            output_directory = workspace / "output"
        else:
            configured_workspace = Path(config.workspace)
            workspace = (
                configured_workspace
                if configured_workspace.is_absolute()
                else base_directory / configured_workspace
            )
            configured_output = Path(config.output.directory)
            output_directory = (
                configured_output
                if configured_output.is_absolute()
                else base_directory / configured_output
            )
        return cls(config, workspace, output_directory=output_directory)

    def run(self) -> RunResult:
        events: list[CanonicalEvent] = []
        for source_config in self.config.sources:
            source = build_source(source_config, self.config.analysis)
            events.extend(source.iter_events())
        events.sort(key=lambda event: (event.timestamp, event.event_id))

        tasks = reconstruct_tasks(events)
        signals = extract_deterministic_signals(tasks, events)
        semantic_analyses = []
        if self.semantic_analyzer is not None:
            for task in tasks:
                analysis = self.semantic_analyzer.analyze(task, events)
                semantic_analyses.append(analysis)
                task.task_type = analysis.task_type
                task.intent = analysis.intent
                for candidate in analysis.signals:
                    signal = OutcomeSignal(
                        signal_id=stable_signal_id(
                            task.task_id,
                            candidate.kind,
                            candidate.subtype,
                            candidate.evidence_event_ids,
                        ),
                        task_id=task.task_id,
                        kind=candidate.kind,
                        subtype=candidate.subtype,
                        polarity=candidate.polarity,
                        confidence=candidate.confidence,
                        evidence_event_ids=candidate.evidence_event_ids,
                        evidence_quotes=candidate.evidence_quotes,
                        source="llm",
                    )
                    signals.append(signal)
                    task.outcome_signals.append(signal)
        metrics = compute_metrics(tasks, signals, group_by=self.config.metrics.group_by)
        proposals = build_proposals(
            tasks, signals, semantic_analyses=semantic_analyses,
        )
        output_directory = self.output_directory

        result = RunResult(
            event_count=len(events),
            task_count=len(tasks),
            events=events,
            tasks=tasks,
            signals=signals,
            semantic_analyses=semantic_analyses,
            metrics=metrics,
            proposals=proposals,
            output_directory=output_directory,
        )
        store = DuckDBStore(self.workspace / "loop.duckdb")
        try:
            store.persist(events, tasks, signals)
        finally:
            store.close()
        write_outputs(result, output_directory, self.config.output.formats)
        return result
