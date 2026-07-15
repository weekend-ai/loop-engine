from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Literal

from loop_engine.models import (
    AssetExposure,
    CanonicalEvent,
    ContextComponent,
    OperationalInvocation,
    TaskRun,
)


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


def _merge_exposures(
    ordered: list[CanonicalEvent],
    marker_exposures: list[AssetExposure],
) -> list[AssetExposure]:
    """Merge marker-based exposures with MCP/plugin/skill from events.

    - MCP servers: state=invoked, version=unknown.
    - Plugins: state=present, version=unknown.
    - Skills: state=present, version=unknown.
    - Tool use upgrades an asset to invoked.
    - Dedup by asset_name; merge evidence event IDs.
    """
    by_name: dict[str, AssetExposure] = {}

    for exp in marker_exposures:
        if exp.asset_name in by_name:
            existing = by_name[exp.asset_name]
            for eid in exp.evidence_event_ids:
                if eid not in existing.evidence_event_ids:
                    existing.evidence_event_ids.append(eid)
            if exp.state == "invoked":
                existing.state = "invoked"
        else:
            by_name[exp.asset_name] = AssetExposure(
                asset_name=exp.asset_name,
                version=exp.version,
                state=exp.state,
                evidence_event_ids=list(exp.evidence_event_ids),
            )

    for event in ordered:
        if event.mcp_server:
            name = f"mcp:{event.mcp_server}"
            if name in by_name:
                exp = by_name[name]
                exp.state = "invoked"
                if event.event_id not in exp.evidence_event_ids:
                    exp.evidence_event_ids.append(event.event_id)
            else:
                by_name[name] = AssetExposure(
                    asset_name=name,
                    version="unknown",
                    state="invoked",
                    evidence_event_ids=[event.event_id],
                )

        if event.plugin_name:
            name = f"plugin:{event.plugin_name}"
            if name in by_name:
                if event.event_id not in by_name[name].evidence_event_ids:
                    by_name[name].evidence_event_ids.append(event.event_id)
                if event.event_type == "tool_use":
                    by_name[name].state = "invoked"
            else:
                state: Literal["present", "invoked"] = (
                    "invoked" if event.event_type == "tool_use"
                    else "present"
                )
                by_name[name] = AssetExposure(
                    asset_name=name,
                    version="unknown",
                    state=state,
                    evidence_event_ids=[event.event_id],
                )

        if event.attribution_skill:
            name = f"skill:{event.attribution_skill}"
            if name in by_name:
                if event.event_id not in by_name[name].evidence_event_ids:
                    by_name[name].evidence_event_ids.append(event.event_id)
            else:
                by_name[name] = AssetExposure(
                    asset_name=name,
                    version="unknown",
                    state="present",
                    evidence_event_ids=[event.event_id],
                )

    return sorted(by_name.values(), key=lambda e: e.asset_name)


def _dedup_invocations(
    invocations: list[OperationalInvocation],
) -> list[OperationalInvocation]:
    """Deduplicate invocations by invocation_id."""
    seen: dict[str, OperationalInvocation] = {}
    for inv in invocations:
        if inv.invocation_id not in seen:
            seen[inv.invocation_id] = inv
    return list(seen.values())


def _dedup_components(
    components: list[ContextComponent],
) -> list[ContextComponent]:
    """Deduplicate context components by (kind, name)."""
    seen: dict[tuple[str, str | None], ContextComponent] = {}
    for comp in components:
        key = (comp.kind, comp.name)
        if key not in seen:
            seen[key] = comp
    return list(seen.values())


def reconstruct_tasks(events: list[CanonicalEvent]) -> list[TaskRun]:
    groups: dict[str, list[CanonicalEvent]] = defaultdict(list)
    for event in events:
        groups[event.session_hint or event.event_id].append(event)

    tasks: list[TaskRun] = []
    for session_id, session_events in sorted(groups.items()):
        ordered = sorted(
            session_events,
            key=lambda event: (event.timestamp, event.event_id),
        )
        user_events = [e for e in ordered if e.role == "user"]
        intent = user_events[0].content if user_events else None

        # Asset markers → exposures
        markers: dict[tuple[str, str], list[str]] = defaultdict(list)
        for event in ordered:
            for marker in event.asset_markers:
                markers[_parse_marker(marker)].append(event.event_id)
        marker_exposures = [
            AssetExposure(
                asset_name=name,
                version=version,
                evidence_event_ids=evidence,
            )
            for (name, version), evidence in sorted(markers.items())
        ]
        exposures = _merge_exposures(ordered, marker_exposures)

        # Token totals — deduplicated by message_id
        input_totals: dict[str, int] = {}
        output_totals: dict[str, int] = {}
        cache_creation_totals: dict[str, int] = {}
        cache_read_totals: dict[str, int] = {}
        for event in ordered:
            msg_key = event.message_id or event.event_id
            if event.input_tokens is not None:
                input_totals[msg_key] = event.input_tokens
            if event.output_tokens is not None:
                output_totals[msg_key] = event.output_tokens
            if event.cache_creation_input_tokens is not None:
                cache_creation_totals[msg_key] = (
                    event.cache_creation_input_tokens
                )
            if event.cache_read_input_tokens is not None:
                cache_read_totals[msg_key] = (
                    event.cache_read_input_tokens
                )

        # HTTP statuses and stop reasons from events
        http_statuses = sorted({
            e.http_status for e in ordered if e.http_status is not None
        })
        stop_reasons_from_events = sorted({
            e.stop_reason for e in ordered if e.stop_reason is not None
        })

        cost_values = [
            e.cost_usd for e in ordered if e.cost_usd is not None
        ]
        latency_values = [
            e.latency_ms for e in ordered if e.latency_ms is not None
        ]

        # Aggregate invocations, components, coverage from events
        all_invocations: list[Any] = []
        all_components: list[Any] = []
        all_coverage_used: list[str] = []
        all_coverage_skipped: list[str] = []
        all_coverage_unresolved: list[str] = []
        for event in ordered:
            all_invocations.extend(event.invocations)
            all_components.extend(event.context_components)
            all_coverage_used.extend(event.coverage_artifacts_used)
            all_coverage_skipped.extend(
                event.coverage_artifacts_skipped
            )
            all_coverage_unresolved.extend(
                event.coverage_unresolved_fields
            )

        # Deduplicate invocations by invocation_id
        deduped_invocations = _dedup_invocations_raw(all_invocations)
        deduped_components = _dedup_components_raw(all_components)

        # When invocations exist, derive totals from them
        if deduped_invocations:
            inv_input = sum(
                inv.get("input_tokens") or 0
                for inv in deduped_invocations
            )
            inv_output = sum(
                inv.get("output_tokens") or 0
                for inv in deduped_invocations
            )
            inv_cache_create = sum(
                inv.get("cache_creation_input_tokens") or 0
                for inv in deduped_invocations
            )
            inv_cache_read = sum(
                inv.get("cache_read_input_tokens") or 0
                for inv in deduped_invocations
            )
            inv_latency = sum(
                inv.get("latency_ms") or 0
                for inv in deduped_invocations
            )
            final_input = inv_input
            final_output = inv_output
            final_cache_create = inv_cache_create
            final_cache_read = inv_cache_read
            final_latency = inv_latency or None
            # Derive HTTP/stop from invocations
            inv_http: list[int] = sorted(
                int(inv["http_status"])
                for inv in deduped_invocations
                if inv.get("http_status") is not None
            )
            inv_stop: list[str] = sorted(
                str(inv["stop_reason"])
                for inv in deduped_invocations
                if inv.get("stop_reason") is not None
            )
            final_http = inv_http if inv_http else http_statuses
            final_stop = inv_stop if inv_stop else stop_reasons_from_events
        else:
            final_input = sum(input_totals.values())
            final_output = sum(output_totals.values())
            final_cache_create = sum(
                cache_creation_totals.values()
            )
            final_cache_read = sum(cache_read_totals.values())
            final_latency = (
                sum(latency_values) if latency_values else None
            )
            final_http = http_statuses
            final_stop = stop_reasons_from_events

        tasks.append(
            TaskRun(
                task_id=f"task:{session_id}",
                session_id=session_id,
                event_ids=[e.event_id for e in ordered],
                intent=intent,
                task_type=_task_type(intent),
                started_at=ordered[0].timestamp,
                ended_at=ordered[-1].timestamp,
                model_ids=sorted({
                    e.model for e in ordered if e.model
                }),
                tool_names=sorted({
                    e.tool_name for e in ordered if e.tool_name
                }),
                asset_exposures=exposures,
                input_tokens=final_input,
                output_tokens=final_output,
                cost_usd=(
                    sum(cost_values) if cost_values else None
                ),
                latency_ms=final_latency,
                cache_creation_input_tokens=final_cache_create,
                cache_read_input_tokens=final_cache_read,
                http_statuses=final_http,
                stop_reasons=final_stop,
                invocations=[
                    OperationalInvocation.model_validate(inv)
                    for inv in deduped_invocations
                ],
                context_components=[
                    ContextComponent.model_validate(comp)
                    for comp in deduped_components
                ],
                coverage_artifacts_used=sorted(set(
                    all_coverage_used
                )),
                coverage_artifacts_skipped=sorted(set(
                    all_coverage_skipped
                )),
                coverage_unresolved_fields=sorted(set(
                    all_coverage_unresolved
                )),
                total_artifact_count=(
                    len(set(all_coverage_used))
                    + len(set(all_coverage_skipped))
                ),
            )
        )
    return tasks


def _dedup_invocations_raw(
    invocations: list[Any],
) -> list[dict[str, Any]]:
    """Deduplicate invocations by invocation_id."""
    seen: dict[str, dict[str, Any]] = {}
    for inv in invocations:
        d = inv if isinstance(inv, dict) else inv.model_dump(mode="json")
        iid = d.get("invocation_id", "")
        if iid not in seen:
            seen[iid] = d
    return list(seen.values())


def _dedup_components_raw(
    components: list[Any],
) -> list[dict[str, Any]]:
    """Deduplicate components by (kind, name)."""
    seen: dict[tuple[str, str | None], dict[str, Any]] = {}
    for comp in components:
        d = comp if isinstance(comp, dict) else comp.model_dump(mode="json")
        key = (d.get("kind", ""), d.get("name"))
        if key not in seen:
            seen[key] = d
    return list(seen.values())
