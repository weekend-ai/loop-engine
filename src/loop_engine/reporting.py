from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loop_engine.models import RunResult
from loop_engine.security import secure_directory, secure_write_text


def _dump(path: Path, value: Any) -> None:
    secure_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    )


def _format_group(group: dict[str, str]) -> str:
    return "; ".join(f"{key}={value}" for key, value in group.items()) or "all"


def write_outputs(result: RunResult, output_dir: Path, formats: Sequence[str]) -> None:
    secure_directory(output_dir)
    if "json" in formats:
        _dump(
            output_dir / "task_runs.json", [task.model_dump(mode="json") for task in result.tasks]
        )
        _dump(
            output_dir / "signals.json",
            [signal.model_dump(mode="json") for signal in result.signals],
        )
        _dump(
            output_dir / "semantic_analyses.json",
            [analysis.model_dump(mode="json") for analysis in result.semantic_analyses],
        )
        _dump(
            output_dir / "metrics.json",
            [metric.model_dump(mode="json") for metric in result.metrics],
        )
        _dump(
            output_dir / "improvement_proposals.json",
            [proposal.model_dump(mode="json") for proposal in result.proposals],
        )
    if "markdown" in formats:
        metric_lines = [
            f"| {metric.name} | {_format_group(metric.group)} | "
            f"{metric.value if metric.value is not None else 'n/a'} | "
            f"{metric.coverage:.0%} | {metric.confidence} |"
            for metric in result.metrics
        ]
        proposal_lines = [
            f"- **{proposal.title}** — {proposal.hypothesis}" for proposal in result.proposals
        ] or ["- No negative evidence-backed pattern met the proposal threshold."]
        report = "\n".join(
            [
                "# AI Learning Loop Report",
                "",
                f"- Events: **{result.event_count}**",
                f"- Task runs: **{result.task_count}**",
                f"- Outcome signals: **{len(result.signals)}**",
                "",
                "## Metrics",
                "",
                "| Metric | Group | Value | Coverage | Confidence |",
                "|---|---|---:|---:|---|",
                *metric_lines,
                "",
                "## Improvement proposals",
                "",
                *proposal_lines,
                "",
                "## Evidence contract",
                "",
                "LLM-derived and deterministic signals retain event IDs and quotes. "
                "A proposal is a hypothesis until validated by a registered experiment.",
                "",
            ]
        )
        secure_write_text(output_dir / "report.md", report)
