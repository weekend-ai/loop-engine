from __future__ import annotations

import json
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
    tool_arguments_json: str | None = None
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
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    http_status: int | None = None
    stop_reason: str | None = None
    raw_ref: str

    @property
    def tool_arguments(self) -> dict[str, Any] | None:
        """Parse tool_arguments_json for backward compatibility."""
        if self.tool_arguments_json is None:
            return None
        try:
            parsed = json.loads(self.tool_arguments_json)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None


class RawRecordEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    source_id: str
    record_id: str
    raw_ref: str
    line_number: int = Field(ge=1)
    raw: Any


class CanonicalEventCandidate(BaseModel):
    """Full normalized candidate after deterministic enrichment from envelope.

    This is the internal analytical model — NOT what the LLM returns.
    The LLM returns LlmNormalizationCandidate (portable closed schema);
    envelope facts (timestamp, tokens, session, etc.) are joined back
    deterministically in finalize_candidates.
    """

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
    tool_arguments_json: str | None = None
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
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    http_status: int | None = None
    stop_reason: str | None = None

    @model_validator(mode="after")
    def validate_tool_contract(self) -> CanonicalEventCandidate:
        if self.event_type == "tool_result" and self.role != "tool":
            raise ValueError("tool_result candidates require role='tool'")
        if self.event_type == "tool_use" and not self.tool_name:
            raise ValueError("tool_use candidates require tool_name")
        return self

    @property
    def tool_arguments(self) -> dict[str, Any] | None:
        """Parse tool_arguments_json for backward compatibility."""
        if self.tool_arguments_json is None:
            return None
        try:
            parsed = json.loads(self.tool_arguments_json)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None


class LlmNormalizationCandidate(BaseModel):
    """Portable closed schema for LLM structured output.

    Contains ONLY interpreted fields the LLM must provide.
    Immutable envelope facts (timestamp, tokens, session_hint, etc.)
    are NOT requested from the LLM — they are joined back from the
    envelope deterministically using record_id + block_index.

    All nested structures use JSON strings instead of open dicts,
    making the schema compatible with both Anthropic and OpenAI
    strict mode (which requires additionalProperties: false).

    STRICT-MODE CONTRACT: every property is listed in 'required'.
    Nullable fields use `str | None` WITHOUT a default so Pydantic
    emits them in the required array. OpenAI strict schemas reject
    any property not in required.
    """

    model_config = ConfigDict(extra="forbid")

    record_id: str
    block_index: int = Field(ge=0)
    event_type: Literal["message", "tool_use", "tool_result", "api_error"]
    role: str | None
    content: str | None
    tool_name: str | None
    tool_arguments_json: str | None = Field(
        description="Tool arguments as a JSON string. Do not use a dict/object."
    )
    tool_result: str | None
    tool_call_id: str | None
    status: Literal["success", "error"] | None
    mcp_server: str | None
    plugin_name: str | None
    attribution_skill: str | None

    @model_validator(mode="after")
    def validate_tool_contract(self) -> LlmNormalizationCandidate:
        if self.event_type == "tool_result" and self.role != "tool":
            raise ValueError("tool_result candidates require role='tool'")
        if self.event_type == "tool_use" and not self.tool_name:
            raise ValueError("tool_use candidates require tool_name")
        return self


class LlmNormalizationBatch(BaseModel):
    """Batch response wrapper for LLM normalization output."""

    model_config = ConfigDict(extra="forbid")

    events: list[LlmNormalizationCandidate]


# ---------------------------------------------------------------------------
# NormalizedTraceBundle — LLM-first raw trace output schema
# ---------------------------------------------------------------------------
# The LLM returns ONE of these per capture bundle. Every extracted fact
# cites the artifact_id it came from via an EvidenceRef. Deterministic
# code validates citations, pairs tools, deduplicates, and converts to
# CanonicalEvents. No provider-specific field parsing in Python.


class EvidenceRef(BaseModel):
    """Citation linking an extracted fact to a source artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    locator: str | None = Field(
        description=(
            "Where in the artifact: JSON path, line number, "
            "stream event index, or null for whole-artifact."
        )
    )


class TraceBundleIdentity(BaseModel):
    """Session/request identity extracted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None
    request_id: str | None
    parent_id: str | None
    evidence: list[EvidenceRef]


class TraceBundleTiming(BaseModel):
    """Timing facts extracted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    start_timestamp: str | None
    end_timestamp: str | None
    latency_ms: int | None
    evidence: list[EvidenceRef]


class TraceBundleUsage(BaseModel):
    """Token/cache usage extracted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None
    output_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    evidence: list[EvidenceRef]


class TraceBundleHTTP(BaseModel):
    """HTTP-level facts extracted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    status_code: int | None
    stop_reason: str | None
    model: str | None
    evidence: list[EvidenceRef]


class TraceToolCall(BaseModel):
    """A single tool call extracted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    tool_name: str
    arguments_json: str | None = Field(
        description="Tool arguments as a JSON string."
    )
    mcp_server: str | None
    plugin_name: str | None
    attribution_skill: str | None
    evidence: list[EvidenceRef]


class TraceToolResult(BaseModel):
    """A single tool result extracted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    content: str | None
    is_error: bool
    evidence: list[EvidenceRef]


class TraceMessage(BaseModel):
    """A message (user, assistant, or system) extracted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str | None
    evidence: list[EvidenceRef]


class TraceCoverage(BaseModel):
    """LLM's self-report of normalization coverage."""

    model_config = ConfigDict(extra="forbid")

    artifacts_used: list[str] = Field(
        description="artifact_ids the LLM actually extracted facts from."
    )
    artifacts_skipped: list[str] = Field(
        description="artifact_ids the LLM could not interpret."
    )
    unresolved_fields: list[str] = Field(
        description=(
            "Field names or paths the LLM saw but could not "
            "map to the schema."
        )
    )


class OperationalInvocation(BaseModel):
    """Per-call telemetry for a model invocation within a trace."""

    model_config = ConfigDict(extra="forbid")

    invocation_id: str
    model: str | None
    start_timestamp: str | None
    end_timestamp: str | None
    latency_ms: int | None
    http_status: int | None
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    thinking_tokens: int | None
    tier: str | None = Field(
        description="Model tier: e.g. 'sonnet', 'haiku', 'opus'."
    )
    evidence: list[EvidenceRef]


class ContextComponent(BaseModel):
    """A measured section of the context window."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        description=(
            "Type: system_prompt, skill_instructions, tool_definitions, "
            "session_context, harness, messages, or custom."
        )
    )
    name: str | None
    char_count: int | None
    item_count: int | None
    cacheable: bool | None
    summary: str | None
    evidence: list[EvidenceRef]


class CompletenessIssue(BaseModel):
    """A specific gap found during completeness review."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    artifact_id: str
    field: str
    description: str


class CompletenessReview(BaseModel):
    """Result of the structured completeness review pass."""

    model_config = ConfigDict(extra="forbid")

    complete: bool
    issues: list[CompletenessIssue]


class NormalizedTraceBundle(BaseModel):
    """Complete LLM-first normalization of a raw trace bundle.

    The LLM returns one of these per capture bundle. Every extracted
    fact cites its source artifact via EvidenceRef. Deterministic code
    validates citations, pairs tool calls/results, deduplicates usage,
    and converts to CanonicalEvents.

    STRICT-MODE CONTRACT: all properties required, nullable where
    appropriate, no defaults. Compatible with OpenAI strict schemas.
    """

    model_config = ConfigDict(extra="forbid")

    identity: TraceBundleIdentity
    timing: TraceBundleTiming
    http: TraceBundleHTTP
    usage: TraceBundleUsage
    messages: list[TraceMessage]
    tool_calls: list[TraceToolCall]
    tool_results: list[TraceToolResult]
    pending_tool_calls: list[str] = Field(
        description=(
            "tool_call_ids that have a call but no result in this bundle."
        )
    )
    invocations: list[OperationalInvocation]
    context_components: list[ContextComponent]
    coverage: TraceCoverage


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
    """Legacy signal candidate — kept for backward compatibility."""

    kind: str
    subtype: str | None = None
    polarity: Literal["positive", "negative", "neutral", "unknown"] = "unknown"
    confidence: float = Field(ge=0, le=1)
    evidence_event_ids: list[str] = Field(min_length=1)
    evidence_quotes: list[str] = Field(default_factory=list)


class SemanticFinding(BaseModel):
    """Evidence-backed observation, inefficiency, or recommendation.

    STRICT-MODE CONTRACT: every property is required (no defaults),
    nullable where appropriate. Compatible with OpenAI strict schemas.
    """

    model_config = ConfigDict(extra="forbid")

    category: str = Field(
        description=(
            "Type of finding: context_overhead, cache_utilization, "
            "redundant_instructions, tool_selection, tool_failure, "
            "plugin_gap, incomplete_trace, or custom."
        )
    )
    summary: str
    rationale: str
    target_layer: Literal[
        "context", "prompt", "plugin", "skill", "tool", "model"
    ]
    target_asset: str | None = Field(
        description="Specific asset name when known (e.g. plugin name)."
    )
    expected_benefit: str | None
    confidence: float = Field(ge=0, le=1)
    evidence_event_ids: list[str]
    evidence_quotes: list[str]
    limitations: str | None = Field(
        description="What evidence is missing or what could invalidate this."
    )
    epistemic_status: Literal[
        "observed_fact", "evidence_backed_inference",
        "tentative_recommendation", "unsupported_conclusion",
    ]


class TaskSemanticAnalysis(BaseModel):
    """Expanded semantic analysis output.

    STRICT-MODE CONTRACT: every property is required, no defaults,
    additionalProperties: false. Compatible with OpenAI strict schemas.
    """

    model_config = ConfigDict(extra="forbid")

    task_type: str
    intent: str
    observations: list[SemanticFinding]
    inefficiencies: list[SemanticFinding]
    recommendations: list[SemanticFinding]
    outcome_signals: list[SemanticFinding]
    missing_evidence: list[str]
    root_cause_hypotheses: list[str]
    signals: list[SemanticSignalCandidate]


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
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    http_statuses: list[int] = Field(default_factory=list)
    stop_reasons: list[str] = Field(default_factory=list)
    invocations: list[OperationalInvocation] = Field(default_factory=list)
    context_components: list[ContextComponent] = Field(default_factory=list)
    coverage_artifacts_used: list[str] = Field(default_factory=list)
    coverage_unresolved_fields: list[str] = Field(default_factory=list)
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
    evidence_event_ids: list[str] = Field(default_factory=list)
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
