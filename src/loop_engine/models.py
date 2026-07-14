from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class CanonicalEvent(BaseModel):
    event_id: str
    source_id: str
    timestamp: datetime
    event_type: str
    session_hint: str | None = None
    parent_hint: str | None = None
    actor_id: str | None = None
    role: str | None = None
    content: str | None = None
    content_hash: str | None = None
    model: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: str | None = None
    status: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    asset_markers: list[str] = Field(default_factory=list)
    raw_ref: str


class AssetExposure(BaseModel):
    asset_name: str
    version: str
    state: Literal["present", "invoked"] = "present"
    evidence_event_ids: list[str] = Field(default_factory=list)


class OutcomeSignal(BaseModel):
    signal_id: str
    task_id: str
    kind: str
    subtype: str | None = None
    polarity: Literal["positive", "negative", "neutral", "unknown"] = "unknown"
    confidence: float = Field(ge=0, le=1)
    evidence_event_ids: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    source: Literal["deterministic", "llm", "human"] = "deterministic"


class SemanticSignalCandidate(BaseModel):
    kind: str
    subtype: str | None = None
    polarity: Literal["positive", "negative", "neutral", "unknown"] = "unknown"
    confidence: float = Field(ge=0, le=1)
    evidence_event_ids: list[str] = Field(min_length=1)
    evidence_quotes: list[str] = Field(default_factory=list)


class TaskSemanticAnalysis(BaseModel):
    task_type: str
    intent: str
    signals: list[SemanticSignalCandidate] = Field(default_factory=list)
    root_cause_hypotheses: list[str] = Field(default_factory=list)


class TaskRun(BaseModel):
    task_id: str
    session_id: str
    event_ids: list[str]
    intent: str | None = None
    task_type: str = "unknown"
    started_at: datetime
    ended_at: datetime | None = None
    model_ids: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    asset_exposures: list[AssetExposure] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: int | None = None
    outcome_signals: list[OutcomeSignal] = Field(default_factory=list)
    reconstruction_method: str = "source_session_hint"
    reconstruction_confidence: float = Field(default=1.0, ge=0, le=1)


class MetricResult(BaseModel):
    name: str
    value: float | None
    numerator: float | None = None
    denominator: float | None = None
    coverage: float = Field(ge=0, le=1)
    confidence: Literal["low", "medium", "high"] = "medium"
    group: dict[str, str] = Field(default_factory=dict)
    excluded: int = 0


class ImprovementProposal(BaseModel):
    proposal_id: str
    task_type: str
    title: str
    hypothesis: str
    target_layer: str
    evidence_signal_ids: list[str] = Field(default_factory=list)
    recommended_experiment: str
    status: Literal["proposed", "approved", "rejected", "deployed"] = "proposed"


class ExperimentResult(BaseModel):
    experiment_id: str
    metric_name: str
    baseline_value: float | None
    candidate_value: float | None
    absolute_delta: float | None
    relative_delta: float | None
    baseline_n: int
    candidate_n: int
    verdict: Literal["improved", "regressed", "inconclusive"]


class RunResult(BaseModel):
    event_count: int
    task_count: int
    events: list[CanonicalEvent]
    tasks: list[TaskRun]
    signals: list[OutcomeSignal]
    semantic_analyses: list[TaskSemanticAnalysis] = Field(default_factory=list)
    metrics: list[MetricResult]
    proposals: list[ImprovementProposal]
    output_directory: Path
