from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    message_id: str | None = None
    tool_call_id: str | None = None
    paired_event_id: str | None = None
    mcp_server: str | None = None
    plugin_name: str | None = None
    attribution_skill: str | None = None
    asset_markers: list[str] = Field(default_factory=list)
    raw_ref: str


class RawRecordEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    source_id: str
    record_id: str
    raw_ref: str
    line_number: int = Field(ge=1)
    raw: Any


class CanonicalEventCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    block_index: int = Field(default=0, ge=0)
    timestamp: str
    event_type: Literal["message", "tool_use", "tool_result", "api_error"]
    session_hint: str | None = None
    parent_hint: str | None = None
    actor_id: str | None = None
    role: str | None = None
    content: str | None = None
    model: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: str | None = None
    status: Literal["success", "error"] | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    message_id: str | None = None
    tool_call_id: str | None = None
    mcp_server: str | None = None
    plugin_name: str | None = None
    attribution_skill: str | None = None
    asset_markers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tool_contract(self) -> CanonicalEventCandidate:
        if self.event_type == "tool_result" and self.role != "tool":
            raise ValueError("tool_result candidates require role='tool'")
        if self.event_type == "tool_use" and not self.tool_name:
            raise ValueError("tool_use candidates require tool_name")
        return self


class CanonicalEventCandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[CanonicalEventCandidate] = Field(default_factory=list)


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
