from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loop_engine.models import RunResult, SemanticFinding
from loop_engine.security import secure_directory, secure_write_text


def _dump(path: Path, value: Any) -> None:
    secure_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=str)
        + "\n",
    )


def _format_group(group: dict[str, str]) -> str:
    return (
        "; ".join(f"{key}={value}" for key, value in group.items())
        or "all"
    )


def _findings_section(
    title: str, findings: list[SemanticFinding]
) -> list[str]:
    """Render a list of SemanticFinding as markdown bullet points."""
    if not findings:
        return [f"### {title}", "", "_None identified._", ""]
    lines = [f"### {title}", ""]
    for f in findings:
        lines.append(
            f"- **{f.summary}** ({f.category}, "
            f"{f.epistemic_status.replace('_', ' ')})"
        )
        lines.append(f"  - Layer: `{f.target_layer}`")
        if f.target_asset:
            lines.append(f"  - Asset: `{f.target_asset}`")
        lines.append(f"  - Confidence: {f.confidence:.0%}")
        lines.append(f"  - {f.rationale}")
        if f.expected_benefit:
            lines.append(f"  - Expected benefit: {f.expected_benefit}")
        if f.limitations:
            lines.append(f"  - ⚠ {f.limitations}")
        if f.evidence_quotes:
            lines.append("  - Evidence:")
            for q in f.evidence_quotes[:3]:
                lines.append(f'    > "{q[:200]}"')
        lines.append("")
    return lines


def write_outputs(
    result: RunResult,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    secure_directory(output_dir)

    if "json" in formats:
        _dump(
            output_dir / "task_runs.json",
            [t.model_dump(mode="json") for t in result.tasks],
        )
        _dump(
            output_dir / "signals.json",
            [s.model_dump(mode="json") for s in result.signals],
        )
        _dump(
            output_dir / "semantic_analyses.json",
            [a.model_dump(mode="json") for a in result.semantic_analyses],
        )
        _dump(
            output_dir / "metrics.json",
            [m.model_dump(mode="json") for m in result.metrics],
        )
        _dump(
            output_dir / "improvement_proposals.json",
            [p.model_dump(mode="json") for p in result.proposals],
        )

    if "markdown" in formats:
        lines: list[str] = [
            "# AI Learning Loop Report",
            "",
        ]

        # --- Trace summary ---
        lines.append("## Trace summary")
        lines.append("")
        lines.append(f"- Events: **{result.event_count}**")
        lines.append(f"- Task runs: **{result.task_count}**")
        lines.append(
            f"- Outcome signals: **{len(result.signals)}**"
        )
        for task in result.tasks:
            lines.append(f"- Session: `{task.session_id}`")
            if task.model_ids:
                lines.append(
                    f"- Model: `{', '.join(task.model_ids)}`"
                )
            if task.latency_ms is not None:
                lines.append(
                    f"- Latency: **{task.latency_ms:,} ms**"
                )
        lines.append("")

        # --- Context and token profile ---
        lines.append("## Context and token profile")
        lines.append("")
        for task in result.tasks:
            lines.append(
                "| Metric | Value |\n|---|---:|"
            )
            lines.append(
                f"| Input tokens | {task.input_tokens:,} |"
            )
            lines.append(
                f"| Output tokens | {task.output_tokens:,} |"
            )
            if task.cost_usd is not None:
                lines.append(
                    f"| Cost | ${task.cost_usd:.4f} |"
                )
            if task.latency_ms is not None:
                lines.append(
                    f"| Latency | {task.latency_ms:,} ms |"
                )
        lines.append("")

        # --- Tools, MCP, plugins, skills ---
        lines.append("## Tools, MCP plugins and skills")
        lines.append("")
        for task in result.tasks:
            if task.tool_names:
                lines.append("### Tools used")
                for t in task.tool_names:
                    lines.append(f"- `{t}`")
                lines.append("")
            if task.asset_exposures:
                lines.append("### Asset exposures")
                for exp in task.asset_exposures:
                    state = f" ({exp.state})" if exp.state else ""
                    lines.append(
                        f"- `{exp.asset_name}` "
                        f"v{exp.version}{state} — "
                        f"{len(exp.evidence_event_ids)} event(s)"
                    )
                lines.append("")

        # --- Semantic findings ---
        for analysis in result.semantic_analyses:
            # Pending / incomplete
            pending_events = [
                e for e in result.events
                if e.event_type == "tool_use"
                and e.tool_call_id
                and not e.paired_event_id
            ]
            if pending_events:
                lines.append("## Pending or incomplete activity")
                lines.append("")
                for e in pending_events:
                    lines.append(
                        f"- Tool call `{e.tool_name}` "
                        f"(id: `{e.tool_call_id}`) — "
                        f"no result received"
                    )
                lines.append("")

            lines.extend(
                _findings_section(
                    "Observed inefficiencies",
                    analysis.inefficiencies,
                )
            )
            lines.extend(
                _findings_section(
                    "Recommended improvements",
                    analysis.recommendations,
                )
            )
            lines.extend(
                _findings_section(
                    "Observations",
                    analysis.observations,
                )
            )

            if analysis.missing_evidence:
                lines.append("### Missing evidence and confidence")
                lines.append("")
                for item in analysis.missing_evidence:
                    lines.append(f"- {item}")
                lines.append("")

        # --- Metrics ---
        lines.append("## Metrics")
        lines.append("")
        lines.append(
            "| Metric | Group | Value | Coverage | Confidence |"
        )
        lines.append("|---|---|---:|---:|---|")
        for metric in result.metrics:
            val = (
                metric.value if metric.value is not None else "n/a"
            )
            lines.append(
                f"| {metric.name} | "
                f"{_format_group(metric.group)} | "
                f"{val} | "
                f"{metric.coverage:.0%} | "
                f"{metric.confidence} |"
            )
        lines.append("")

        # --- Proposals ---
        lines.append("## Improvement proposals")
        lines.append("")
        if result.proposals:
            for p in result.proposals:
                lines.append(
                    f"- **{p.title}** "
                    f"(layer: `{p.target_layer}`)"
                )
                lines.append(f"  - {p.hypothesis}")
                lines.append(
                    f"  - Experiment: {p.recommended_experiment}"
                )
                lines.append("")
        else:
            lines.append(
                "- No evidence-backed recommendations met the "
                "proposal threshold."
            )
            lines.append("")

        # --- Evidence contract ---
        lines.append("## Supporting evidence")
        lines.append("")
        lines.append(
            "LLM-derived and deterministic signals retain event "
            "IDs and quotes. A proposal is a hypothesis until "
            "validated by a registered experiment."
        )
        lines.append("")

        secure_write_text(
            output_dir / "report.md", "\n".join(lines)
        )
