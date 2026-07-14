# Architecture Decisions

## ADR-001: Local-first CLI before a service or dashboard

Phase 0 must prove the loop closes. A composable CLI and file artifacts expose each stage, keep raw traces local, and avoid UI/auth/queue work that does not validate the mechanism.

## ADR-002: Deterministic mechanics, model-assisted interpretation

JSON framing, byte limits, redaction, identifiers, timestamp parsing, candidate validation, pairing, reconstruction, costs, ratios, coverage, and experiment deltas are code. Claude may interpret raw provider schema and task semantics, but it returns constrained candidates rather than authoritative IDs or metrics. This prevents an LLM from grading its own numeric claims or controlling trust-boundary mechanics.

## ADR-003: Raw envelopes precede CanonicalEvent

Claude JSONL first becomes a tolerant `RawRecordEnvelope` whose `raw` value retains unknown fields and types. A configured normalizer interprets envelopes into strict candidates; deterministic local finalization creates provider-neutral `CanonicalEvent` rows. The local fallback maps known shapes and skips unsupported values without crashing. LiteLLM remains a direct deterministic adapter. Future OpenTelemetry, Langfuse, or other adapters can adopt either path without rewriting downstream stages.

## ADR-004: Proposals are not improvements

An ImprovementProposal is only a hypothesis. Improvement exists only after an asset version is registered in traces and the experiment evaluator compares a predefined metric and guardrails.

## ADR-005: One session equals one task only in Phase 0

The contracts distinguish Session and TaskRun even though the first reconstruction implementation maps one session to one task. This limitation is explicit and replaceable rather than hidden.

## ADR-006: External semantic analysis is explicit and bounded

Rule-based normalization and analysis are local. Either Claude phase requires explicit data-egress opt-in. Raw normalization preserves record structure but sends only recursively redacted/truncated values and opaque record IDs—never local paths—under per-record and total limits. Semantic analysis sends a canonical field whitelist. Both invoke Claude Code in bare/no-tools/no-session mode with a timeout. These controls reduce accidental exposure but do not replace organizational DLP approval.

## ADR-007: DuckDB is a current snapshot, not an append-only audit log

Each successful persist replaces events, tasks, and signals in one transaction. Stable namespaced IDs make reruns auditable, while deleted source records do not remain indefinitely. Historical retention requires a future run/version model rather than accidental table accumulation.

## ADR-008: Canonical identifiers are stable and unambiguous

Manifest source IDs are unique and restricted to a safe character set. Canonical IDs encode source/raw components with explicit lengths, and missing upstream IDs derive from a hash of the full raw reference. Process memory addresses and ambiguous delimiter concatenation are forbidden.

## ADR-009: Remote ingestion limits apply during reads

S3 prefixes are directory-scoped. Each body read is bounded by the smaller remaining object/run allowance plus one detection byte, so configured limits are enforced before an unbounded object can be retained in memory.
