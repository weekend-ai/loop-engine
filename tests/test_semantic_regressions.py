"""Regression tests for asset evidence accumulation and proposal evidence."""

from __future__ import annotations

from datetime import UTC, datetime

from loop_engine.models import (
    CanonicalEvent,
    SemanticFinding,
    TaskRun,
    TaskSemanticAnalysis,
)
from loop_engine.proposals import build_proposals
from loop_engine.reconstruction import reconstruct_tasks


def _event(
    event_id: str,
    *,
    session: str = "s1",
    event_type: str = "message",
    role: str = "assistant",
    tool_name: str | None = None,
    mcp_server: str | None = None,
    plugin_name: str | None = None,
    attribution_skill: str | None = None,
) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=event_id,
        source_id="test",
        timestamp=datetime(2026, 7, 14, tzinfo=UTC),
        event_type=event_type,
        session_hint=session,
        role=role,
        tool_name=tool_name,
        mcp_server=mcp_server,
        plugin_name=plugin_name,
        attribution_skill=attribution_skill,
        raw_ref=f"test:{event_id}",
    )


# -----------------------------------------------------------------------
# Asset evidence accumulation
# -----------------------------------------------------------------------


def test_mcp_server_exposure_merges_evidence_across_events() -> None:
    """Multiple tool_use events for the same MCP server merge evidence."""
    events = [
        _event(
            "e1",
            event_type="tool_use",
            tool_name="mcp__datahub__search",
            mcp_server="datahub",
        ),
        _event(
            "e2",
            event_type="tool_use",
            tool_name="mcp__datahub__get_schema",
            mcp_server="datahub",
        ),
    ]
    tasks = reconstruct_tasks(events)
    assert len(tasks) == 1
    datahub_exposures = [
        e for e in tasks[0].asset_exposures
        if e.asset_name == "mcp:datahub"
    ]
    assert len(datahub_exposures) == 1
    exp = datahub_exposures[0]
    assert exp.state == "invoked"
    assert exp.version == "unknown"
    assert set(exp.evidence_event_ids) == {"e1", "e2"}


def test_plugin_exposure_upgrades_to_invoked_on_tool_use() -> None:
    """A message event creates 'present'; tool_use upgrades to 'invoked'."""
    events = [
        _event("e1", plugin_name="datahub"),
        _event(
            "e2",
            event_type="tool_use",
            tool_name="mcp__datahub__search",
            plugin_name="datahub",
            mcp_server="datahub",
        ),
    ]
    tasks = reconstruct_tasks(events)
    plugin_exp = [
        e for e in tasks[0].asset_exposures
        if e.asset_name == "plugin:datahub"
    ]
    assert len(plugin_exp) == 1
    assert plugin_exp[0].state == "invoked"
    assert set(plugin_exp[0].evidence_event_ids) == {"e1", "e2"}


def test_skill_exposure_present_state() -> None:
    """Attribution skill creates a 'present' exposure."""
    events = [_event("e1", attribution_skill="mcp-tools")]
    tasks = reconstruct_tasks(events)
    skill_exp = [
        e for e in tasks[0].asset_exposures
        if e.asset_name == "skill:mcp-tools"
    ]
    assert len(skill_exp) == 1
    assert skill_exp[0].state == "present"
    assert skill_exp[0].version == "unknown"


# -----------------------------------------------------------------------
# Proposal evidence: event IDs, not signal IDs
# -----------------------------------------------------------------------


def _make_analysis_with_recommendation(
    evidence_ids: list[str],
) -> TaskSemanticAnalysis:
    return TaskSemanticAnalysis(
        task_type="mcp_tool_use",
        intent="Search DataHub",
        observations=[],
        inefficiencies=[],
        recommendations=[
            SemanticFinding(
                category="context_overhead",
                summary="Reduce system prompt size",
                rationale="96k cache creation tokens suggest oversized context",
                target_layer="context",
                target_asset=None,
                expected_benefit="Reduce cache creation cost",
                confidence=0.8,
                evidence_event_ids=evidence_ids,
                evidence_quotes=["96651 cache_creation_input_tokens"],
                limitations=None,
                epistemic_status="evidence_backed_inference",
            )
        ],
        outcome_signals=[],
        missing_evidence=[],
        root_cause_hypotheses=[],
        signals=[],
    )


def test_proposal_carries_evidence_event_ids_not_signal_ids() -> None:
    """Proposals must use evidence_event_ids, never evidence_signal_ids."""
    task = TaskRun(
        task_id="t1",
        session_id="s1",
        event_ids=["e1"],
        started_at=datetime(2026, 7, 14, tzinfo=UTC),
    )
    analysis = _make_analysis_with_recommendation(["e1"])
    proposals = build_proposals(
        [task], [],
        semantic_analyses=[analysis],
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert hasattr(p, "evidence_event_ids")
    assert not hasattr(p, "evidence_signal_ids")
    assert p.evidence_event_ids == ["e1"]


def test_proposal_from_recommendation_is_specific() -> None:
    """Proposals carry the recommendation's title/rationale, not boilerplate."""
    task = TaskRun(
        task_id="t1",
        session_id="s1",
        event_ids=["e1"],
        started_at=datetime(2026, 7, 14, tzinfo=UTC),
    )
    analysis = _make_analysis_with_recommendation(["e1"])
    proposals = build_proposals(
        [task], [],
        semantic_analyses=[analysis],
    )
    assert proposals[0].title == "Reduce system prompt size"
    assert "cache creation" in proposals[0].hypothesis


def test_proposal_requires_evidence() -> None:
    """Recommendations without evidence_event_ids are not promoted."""
    task = TaskRun(
        task_id="t1",
        session_id="s1",
        event_ids=["e1"],
        started_at=datetime(2026, 7, 14, tzinfo=UTC),
    )
    analysis = _make_analysis_with_recommendation([])  # no evidence
    proposals = build_proposals(
        [task], [],
        semantic_analyses=[analysis],
    )
    assert len(proposals) == 0
