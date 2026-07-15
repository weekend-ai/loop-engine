from __future__ import annotations

import re
from collections import defaultdict

from loop_engine.models import AssetExposure, CanonicalEvent, TaskRun


def _task_type(intent: str | None) -> str:
    text = (intent or "").lower()
    if any(word in text for word in ("test", "pytest", "failing", "bug", "fix")):
        return "coding_debugging"
    if any(word in text for word in ("summarize", "summary", "总结")):
        return "document_summarization"
    return "unknown"


def _parse_marker(marker: str) -> tuple[str, str]:
    match = re.match(r"^(?P<name>.+?)@(?P<version>[^@]+)$", marker)
    if match:
        return match.group("name"), match.group("version")
    return marker, "unknown"


def reconstruct_tasks(events: list[CanonicalEvent]) -> list[TaskRun]:
    groups: dict[str, list[CanonicalEvent]] = defaultdict(list)
    for event in events:
        groups[event.session_hint or event.event_id].append(event)

    tasks: list[TaskRun] = []
    for session_id, session_events in sorted(groups.items()):
        ordered = sorted(session_events, key=lambda event: (event.timestamp, event.event_id))
        user_events = [event for event in ordered if event.role == "user"]
        intent = user_events[0].content if user_events else None

        # Asset markers
        markers: dict[tuple[str, str], list[str]] = defaultdict(list)
        for event in ordered:
            for marker in event.asset_markers:
                markers[_parse_marker(marker)].append(event.event_id)

        exposures: list[AssetExposure] = [
            AssetExposure(
                asset_name=name,
                version=version,
                evidence_event_ids=evidence,
            )
            for (name, version), evidence in sorted(markers.items())
        ]

        # Propagate MCP server, plugin, and skill into AssetExposure
        seen_assets: set[tuple[str, str]] = {
            (e.asset_name, e.version) for e in exposures
        }
        for event in ordered:
            if event.mcp_server:
                key = (f"mcp:{event.mcp_server}", "invoked")
                if key not in seen_assets:
                    seen_assets.add(key)
                    exposures.append(AssetExposure(
                        asset_name=f"mcp:{event.mcp_server}",
                        version="invoked",
                        state="invoked",
                        evidence_event_ids=[event.event_id],
                    ))
                else:
                    for exp in exposures:
                        if (
                            exp.asset_name == f"mcp:{event.mcp_server}"
                            and event.event_id not in exp.evidence_event_ids
                        ):
                            exp.evidence_event_ids.append(event.event_id)
            if event.plugin_name:
                key = (f"plugin:{event.plugin_name}", "present")
                if key not in seen_assets:
                    seen_assets.add(key)
                    exposures.append(AssetExposure(
                        asset_name=f"plugin:{event.plugin_name}",
                        version="present",
                        state="present",
                        evidence_event_ids=[event.event_id],
                    ))
            if event.attribution_skill:
                key = (f"skill:{event.attribution_skill}", "present")
                if key not in seen_assets:
                    seen_assets.add(key)
                    exposures.append(AssetExposure(
                        asset_name=f"skill:{event.attribution_skill}",
                        version="present",
                        state="present",
                        evidence_event_ids=[event.event_id],
                    ))

        cost_values = [event.cost_usd for event in ordered if event.cost_usd is not None]
        latency_values = [event.latency_ms for event in ordered if event.latency_ms is not None]
        input_totals: dict[str, int] = {}
        output_totals: dict[str, int] = {}
        for event in ordered:
            msg_key = event.message_id or event.event_id
            if event.input_tokens is not None:
                input_totals[msg_key] = event.input_tokens
            if event.output_tokens is not None:
                output_totals[msg_key] = event.output_tokens
        tasks.append(
            TaskRun(
                task_id=f"task:{session_id}",
                session_id=session_id,
                event_ids=[event.event_id for event in ordered],
                intent=intent,
                task_type=_task_type(intent),
                started_at=ordered[0].timestamp,
                ended_at=ordered[-1].timestamp,
                model_ids=sorted({event.model for event in ordered if event.model}),
                tool_names=sorted({event.tool_name for event in ordered if event.tool_name}),
                asset_exposures=exposures,
                input_tokens=sum(input_totals.values()),
                output_tokens=sum(output_totals.values()),
                cost_usd=sum(cost_values) if cost_values else None,
                latency_ms=sum(latency_values) if latency_values else None,
            )
        )
    return tasks
