# Enterprise AI Loop Engine

Local-first Phase 0 engine for turning AI-agent logs into evidence-backed task metrics, improvement hypotheses, and asset-version experiments.

## What it does

```text
Configured log sources
  -> CanonicalEvent
  -> Session/Task reconstruction
  -> deterministic + Claude semantic signals
  -> aggregate/grouped metrics
  -> improvement proposals
  -> v1/v2 experiment evaluation
  -> DuckDB + JSON + Markdown reports
```

The engine keeps analysis mechanics separate from data. Log locations, analyzer, grouping dimensions, and output formats are runtime configuration.

## Supported inputs

- Claude Code JSONL
  - JSONL is first decoded into tolerant raw envelopes; no Claude field types are assumed at this boundary.
  - `normalizer: rule_based` maps known shapes locally and skips unsupported fields without crashing.
  - `normalizer: claude_sdk` sends bounded, recursively redacted raw records to Claude and validates returned event candidates.
  - Local finalization owns IDs, UTC timestamps, source namespaces, call/result pairing, and validation.
  - Canonical events retain MCP server, plugin/skill attribution, tool call IDs, and paired event IDs.
- Local LiteLLM request JSON/JSONL
- S3 LiteLLM request JSON/JSONL

LiteLLM records with explicit `metadata.session_id` use it. Otherwise, the adapter reconstructs lineage from actor/model plus exact message-history prefixes and emits only the newly-added messages. Timestamps are normalized to UTC; timestamps without an offset are explicitly interpreted as UTC. When a source event/request ID is absent, the fallback is a stable hash of its full raw reference rather than process memory identity.

Source IDs must be unique and match `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`. Canonical event/session IDs use length-prefixed component encoding, so source and raw IDs cannot collide through delimiter ambiguity.

## Install

```bash
cd /root/loop-engine
uv sync
uv run loop --help
```

Python 3.12+ and `uv` are required. Claude semantic analysis additionally requires Anthropic API access. Set `ANTHROPIC_API_KEY` (or `LITELLM_API_KEY`/`LITELLM_MASTER_KEY` with `ANTHROPIC_BASE_URL`/`LITELLM_BASE_URL` for a LiteLLM proxy). Verify with a real request:

```python
import anthropic; print(anthropic.Anthropic().messages.create(model="sonnet", max_tokens=10, messages=[{"role": "user", "content": "OK"}]).content[0].text)
```

## Quick start

Run the deterministic example:

```bash
uv run loop validate -c examples/loop.yaml
uv run loop run -c examples/loop.yaml --workspace .demo
```

Outputs:

```text
.demo/
├── loop.duckdb
└── output/
    ├── task_runs.json
    ├── signals.json
    ├── semantic_analyses.json
    ├── metrics.json
    ├── improvement_proposals.json
    └── report.md
```

Switch only Claude ingestion normalization to Claude (semantic analysis remains local):

```yaml
sources:
  - id: claude-local
    type: claude_code_jsonl
    path: /home/user/.claude/projects/**/*.jsonl
    normalizer: claude_sdk
analysis:
  provider: rule_based
  model: sonnet
  timeout_seconds: 120
  max_input_chars: 100000
  max_event_chars: 4000
  external_data_egress_allowed: true
```

Switch post-reconstruction semantic analysis to Claude:

```yaml
analysis:
  provider: claude_sdk
  model: sonnet
  timeout_seconds: 120
  max_input_chars: 100000
  max_event_chars: 4000
  external_data_egress_allowed: true
```

Claude receives one canonical task bundle at a time and must return strict structured output. Every semantic signal must cite at least one valid event ID. Unknown IDs make the run fail rather than silently creating unsupported evidence.

**Data boundary:** both `rule_based` semantic analysis and `normalizer: rule_based` stay local. Either Claude mode requires `external_data_egress_allowed: true`. Claude ingestion receives the complete raw record structure after recursive key/value credential redaction and per-string truncation, but never receives local `raw_ref` paths. Records are greedily batched under the total input limit and sent via the Anthropic SDK with structured JSON schema output, a timeout, and max retries. Claude can only return strict event candidates referring to supplied record IDs; deterministic local code generates final IDs, parses UTC timestamps, validates duplicate block identities, and pairs tool calls/results. Redaction is defense in depth, not a complete DLP system. If both source normalization and semantic analysis use Claude, the engine intentionally makes two separate provider phases.

## Configuration

```yaml
version: 1
workspace: ./.loop-engine

sources:
  - id: claude-local
    type: claude_code_jsonl
    path: /home/user/.claude/projects/**/*.jsonl
    normalizer: rule_based # or claude_sdk
    max_object_bytes: 10485760 # per raw JSONL record
    max_total_bytes: 104857600

  - id: litellm-local
    type: litellm_local_json
    path: /var/log/litellm/**/*.json

  - id: litellm-s3
    type: litellm_s3_json
    uri: s3://company-litellm/logs/
    aws_profile: readonly
    max_object_bytes: 10485760
    max_total_bytes: 104857600

analysis:
  provider: claude_sdk # or rule_based
  model: sonnet
  max_concurrency: 4
  timeout_seconds: 120
  max_input_chars: 100000
  max_event_chars: 4000
  external_data_egress_allowed: true # required for either claude_sdk phase

metrics:
  group_by: [task_type, model, asset_version]

output:
  directory: ./output
  formats: [json, markdown]
```

Do not put AWS credentials in YAML. Use a read-only AWS profile or normal workload identity. S3 URIs are treated as directory prefixes. The adapter passes an explicit byte bound to every body read and enforces both per-object and per-run limits before retaining additional bytes in memory.

## Metrics contract

Metrics are deterministic. Claude does not calculate aggregates.

Built-ins:

- `task_count`
- `correction_rate`
- `tool_failure_rate`
- `api_failure_rate`
- `objective_success_rate`
- `cost_per_task`
- `latency_per_task_ms`
- `events_per_task`

Every metric includes `numerator`, `denominator`, `coverage`, `confidence`, exclusions, and grouping dimensions. Unknown objective outcomes are excluded rather than counted as failures. Missing cost/latency observations are excluded rather than converted to zero; an explicitly observed zero remains valid. A low-coverage metric is never presented as if it described all tasks.

## Experiment evaluation

After distributing an asset candidate and collecting both versions:

```bash
uv run loop experiment-evaluate \
  --tasks .demo/output/task_runs.json \
  --experiment-id EXP-001 \
  --task-type coding_debugging \
  --asset systematic-debugging \
  --baseline v1 \
  --candidate v2 \
  --metric correction_rate \
  --output .demo/output/EXP-001.json
```

Supported experiment metrics:

- `correction_rate` (lower is better)
- `tool_failure_rate` (lower is better)
- `objective_success_rate` (higher is better)

The result reports baseline/candidate values, observable sample counts, absolute/relative deltas, and a directional verdict. For `objective_success_rate`, unknown outcomes are excluded from each cohort. Tasks exposed to both versions are rejected to prevent cohort contamination. It does not claim statistical or causal certainty.

## Evidence and safety model

- Deterministic and LLM signal IDs are stable hashes of task, kind, subtype, and evidence.
- Deterministic and LLM signals retain source event IDs.
- Claude signals also retain short evidence quotes and confidence.
- Proposals remain hypotheses until evaluated through an experiment.
- The Claude adapter uses the Anthropic SDK with structured JSON schema output, a timeout, and max retries.
- Raw source references remain attached to canonical events for local audit but are omitted from Claude bundles.
- No automatic Skill/Prompt/KB mutation or deployment occurs in Phase 0.

The DuckDB state and JSON/Markdown artifacts can contain canonical event content. The engine creates its dedicated workspace/output directories as `0700` and generated files as `0600`; still use an access-controlled local filesystem. DuckDB represents the latest successful input snapshot: each persist transaction replaces prior events, tasks, and signals. Removing a source record removes it from the next successful snapshot; no long-term retention or purge scheduler is implied.

## Architecture

```text
src/loop_engine/
├── sources/              # Claude JSONL and LiteLLM local/S3 adapters
├── analyzers/            # Claude CLI semantic analyzer
├── models.py             # Canonical contracts
├── reconstruction.py     # Session -> Phase 0 TaskRun
├── signals.py            # deterministic outcome signals
├── metrics.py            # aggregate and grouped metrics
├── proposals.py          # evidence-backed hypotheses
├── experiments.py        # deterministic v1/v2 comparison
├── storage.py            # DuckDB persistence
├── reporting.py          # JSON/Markdown artifacts
├── engine.py             # orchestration
└── cli.py                # Typer CLI
```

## Known Phase 0 limits

- A reconstructed session is treated as one TaskRun; multi-task session segmentation is not automated yet.
- Exact-prefix reconstruction does not bridge context compaction.
- Parallel branches share a session but are not yet represented as a full DAG in reports.
- Silence is never treated as success or abandonment.
- Before/after results may be confounded; use session-level switchback where possible.
- `max_concurrency` is reserved for the next batch-analysis iteration; Phase 0 semantic analysis is sequential.

## Development

```bash
uv run pytest
uv run ruff check .
uv run mypy
```
