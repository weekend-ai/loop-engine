from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from loop_engine.config import load_config
from loop_engine.metrics import compute_metrics
from loop_engine.models import CanonicalEventCandidate, LlmNormalizationCandidate, RawRecordEnvelope
from loop_engine.providers.base import ProviderResponse
from loop_engine.reconstruction import reconstruct_tasks
from loop_engine.signals import extract_deterministic_signals
from loop_engine.sources.claude_jsonl import ClaudeCodeJsonlSource
from loop_engine.sources.claude_normalization import (
    ClaudeSdkRecordNormalizer,
    finalize_candidates,
)


class FakeProvider:
    """Test provider that returns a canned JSON response."""

    def __init__(self, response: dict[str, Any] | str | Exception) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def request_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        payload: str,
        schema: dict[str, Any],
        max_output_tokens: int,
        operation: str,
    ) -> ProviderResponse:
        self.calls.append({
            "model": model,
            "system": system_prompt,
            "payload": payload,
            "schema": schema,
            "max_output_tokens": max_output_tokens,
            "operation": operation,
        })
        if isinstance(self._response, Exception):
            raise self._response
        if isinstance(self._response, str):
            return ProviderResponse(raw_text=self._response)
        return ProviderResponse(raw_text=json.dumps(self._response))


def test_raw_envelope_accepts_unknown_json_types_and_fallback_skips_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tolerant.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps("string record"),
                json.dumps(["future", {"shape": True}]),
                json.dumps({"timestamp": 123, "message": "not-an-object"}),
                json.dumps({"timestamp": "not-an-iso-time", "message": {}}),
            ]
        )
    )
    source = ClaudeCodeJsonlSource("claude", str(path))

    envelopes = list(source.iter_envelopes())
    events = list(source.iter_events())

    assert [type(envelope.raw) for envelope in envelopes] == [str, list, dict, dict]
    assert events == []


def test_sanitized_session_fallback_handles_tools_pairing_attribution_and_usage() -> None:
    path = Path("tests/fixtures/claude_sessions/synthetic-current-schema.jsonl")

    events = list(ClaudeCodeJsonlSource("claude", str(path)).iter_events())
    tasks = reconstruct_tasks(events)
    signals = extract_deterministic_signals(tasks, events)
    metrics = compute_metrics(tasks, signals)

    assert len(tasks) == 1
    assert tasks[0].tool_names == ["Bash", "mcp__github__search"]
    assert tasks[0].input_tokens == 9953
    assert tasks[0].output_tokens == 591
    assert any(event.mcp_server == "github" for event in events)
    assert any(event.plugin_name == "superpowers" for event in events)
    assert any(event.attribution_skill == "systematic-debugging" for event in events)
    paired = [event for event in events if event.tool_call_id and event.paired_event_id]
    assert {event.tool_call_id for event in paired} == {"toolu_mcp", "toolu_bash"}
    assert {event.status for event in events if event.event_type == "tool_result"} == {
        "success",
        "error",
    }
    assert not any(signal.kind == "human_correction" for signal in signals)
    tool_failure = next(metric for metric in metrics if metric.name == "tool_failure_rate")
    assert tool_failure.value == 1.0


def _llm_candidate(**overrides: Any) -> dict[str, Any]:
    """Build a complete LlmNormalizationCandidate dict with all required fields."""
    base: dict[str, Any] = {
        "record_id": "record",
        "block_index": 0,
        "event_type": "message",
        "role": None,
        "content": None,
        "tool_name": None,
        "tool_arguments_json": None,
        "tool_result": None,
        "tool_call_id": None,
        "status": None,
        "mcp_server": None,
        "plugin_name": None,
        "attribution_skill": None,
    }
    base.update(overrides)
    return base


def test_claude_normalizer_redacts_preserves_unknown_fields_and_finalizes_pairing() -> None:
    # LLM returns only interpreted fields — no timestamp, tokens, session_hint
    llm_response = {
        "events": [
            _llm_candidate(
                record_id="record-1",
                event_type="tool_use",
                role="assistant",
                tool_name="mcp__github__search",
                tool_arguments_json='{"query": "safe"}',
                tool_call_id="toolu_1",
                mcp_server="github",
                plugin_name="plugin-x",
                attribution_skill="skill-y",
            ),
            _llm_candidate(
                record_id="record-2",
                event_type="tool_result",
                role="tool",
                tool_result="done",
                content="done",
                status="success",
                tool_call_id="toolu_1",
            ),
        ]
    }
    fake_provider = FakeProvider(llm_response)

    envelopes = [
        RawRecordEnvelope(
            source_id="claude",
            record_id="record-1",
            raw_ref="file:///tmp/session#line=1",
            line_number=1,
            raw={
                "timestamp": "2026-07-14T08:00:00Z",
                "sessionId": "session-1",
                "future_field": {"nested": [1, 2, 3]},
                "password": "RAWSECRET",
                "message": {},
            },
        ),
        RawRecordEnvelope(
            source_id="claude",
            record_id="record-2",
            raw_ref="file:///tmp/session#line=2",
            line_number=2,
            raw={
                "timestamp": "2026-07-14T08:00:01Z",
                "sessionId": "session-1",
                "toolUseResult": "done",
                "message": {},
            },
        ),
    ]
    normalizer = ClaudeSdkRecordNormalizer(provider=fake_provider, repair=False)

    candidates = normalizer.normalize(envelopes)
    events = finalize_candidates(envelopes, candidates)

    # Verify payload sent to provider
    call = fake_provider.calls[0]
    payload = call["payload"]
    assert "future_field" in payload
    assert "RAWSECRET" not in payload
    assert "[REDACTED]" in payload
    assert "file:///tmp/session" not in payload
    # Verify LLM schema is the portable LlmNormalizationBatch
    schema = call["schema"]
    assert "LlmNormalizationCandidate" in json.dumps(schema)

    # Verify deterministic enrichment: envelope facts joined back
    assert events[0].session_hint is not None  # from envelope
    assert events[0].paired_event_id == events[1].event_id
    assert events[1].paired_event_id == events[0].event_id
    assert events[0].mcp_server == "github"
    assert events[0].plugin_name == "plugin-x"
    assert events[0].tool_arguments_json == '{"query": "safe"}'
    assert events[0].tool_arguments == {"query": "safe"}


def test_claude_normalizer_rejects_unknown_record_ids() -> None:
    llm_response = {
        "events": [
            _llm_candidate(record_id="invented"),
        ]
    }
    fake_provider = FakeProvider(llm_response)

    envelope = RawRecordEnvelope(
        source_id="claude",
        record_id="known",
        raw_ref="file:///tmp/session#line=1",
        line_number=1,
        raw={"timestamp": "2026-07-14T08:00:00Z", "message": {}},
    )

    with pytest.raises(RuntimeError, match="unknown record ID"):
        ClaudeSdkRecordNormalizer(provider=fake_provider, repair=False).normalize([envelope])


def test_claude_jsonl_and_normalizer_limits_fail_before_external_call(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large.jsonl"
    path.write_text(json.dumps({"payload": "x" * 200}))
    source = ClaudeCodeJsonlSource(
        "claude", str(path), max_record_bytes=20, max_total_bytes=100
    )
    with pytest.raises(ValueError, match="record size limit"):
        list(source.iter_envelopes())

    fake_provider = FakeProvider({"events": []})

    envelope = RawRecordEnvelope(
        source_id="claude",
        record_id="large",
        raw_ref="file:///tmp/large#line=1",
        line_number=1,
        raw={"unknown": "x" * 500},
    )
    with pytest.raises(RuntimeError, match="exceeds configured limit"):
        ClaudeSdkRecordNormalizer(
            provider=fake_provider, max_input_chars=100, max_record_chars=1000, repair=False
        ).normalize([envelope])
    assert fake_provider.calls == []


def test_canonical_candidate_enforces_tool_contract() -> None:
    with pytest.raises(ValidationError, match="role='tool'"):
        CanonicalEventCandidate(
            record_id="record",
            timestamp="2026-07-14T08:00:00Z",
            event_type="tool_result",
            role="user",
        )
    with pytest.raises(ValidationError, match="require tool_name"):
        CanonicalEventCandidate(
            record_id="record",
            timestamp="2026-07-14T08:00:00Z",
            event_type="tool_use",
            role="assistant",
        )


def test_llm_normalization_candidate_enforces_tool_contract() -> None:
    """LlmNormalizationCandidate has its own tool_contract validator."""
    with pytest.raises(ValidationError, match="role='tool'"):
        LlmNormalizationCandidate(**_llm_candidate(
            event_type="tool_result",
            role="user",
        ))
    with pytest.raises(ValidationError, match="require tool_name"):
        LlmNormalizationCandidate(**_llm_candidate(
            event_type="tool_use",
            role="assistant",
        ))


def test_llm_normalization_candidate_uses_json_string_for_args() -> None:
    """tool_arguments_json is a str, not a dict — portable across providers."""
    candidate = LlmNormalizationCandidate(**_llm_candidate(
        event_type="tool_use",
        role="assistant",
        tool_name="Bash",
        tool_arguments_json='{"cmd": "ls"}',
    ))
    assert candidate.tool_arguments_json == '{"cmd": "ls"}'


def test_claude_source_normalizer_requires_egress_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text(
        "version: 1\n"
        "sources:\n"
        "  - id: claude\n"
        "    type: claude_code_jsonl\n"
        "    path: ./session.jsonl\n"
        "    normalizer: claude_sdk\n"
    )

    with pytest.raises(ValidationError, match="external_data_egress_allowed"):
        load_config(path)

    path.write_text(
        path.read_text()
        + "analysis:\n"
        + "  external_data_egress_allowed: true\n"
    )
    assert load_config(path).sources[0].normalizer == "claude_sdk"


def test_bounded_repair_retries_once_on_validation_failure() -> None:
    """If provider returns bad JSON, repair sends it back once."""

    class TwoResponseProvider:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self._call_count = 0

        def request_structured(self, **kwargs: Any) -> ProviderResponse:
            self.calls.append(kwargs)
            self._call_count += 1
            if self._call_count == 1:
                # First response: bad structure (missing required fields)
                return ProviderResponse(raw_text='{"events": [{"record_id": "r1"}]}')
            # Repair response: correct — all required fields present
            return ProviderResponse(raw_text=json.dumps({
                "events": [_llm_candidate(
                    record_id="r1",
                    role="assistant",
                    content="hello",
                )]
            }))

    provider = TwoResponseProvider()
    envelope = RawRecordEnvelope(
        source_id="claude",
        record_id="r1",
        raw_ref="file:///tmp/x#line=1",
        line_number=1,
        raw={"timestamp": "2026-07-14T08:00:00Z", "message": {}},
    )

    normalizer = ClaudeSdkRecordNormalizer(provider=provider, repair=True)
    candidates = normalizer.normalize([envelope])

    assert len(candidates) == 1
    assert candidates[0].event_type == "message"
    assert len(provider.calls) == 2  # original + repair
    assert "repair" in provider.calls[1]["operation"]


def _check_openai_strict_compliance(
    schema: dict[str, Any],
    path: str = "root",
    defs: dict[str, Any] | None = None,
) -> list[str]:
    """Recursively validate a JSON schema against OpenAI strict-mode rules.

    Rules:
    - Every object must have additionalProperties: false
    - Every property key must appear in required
    - No default values (properties must always be populated)
    - $ref targets must also be compliant
    """
    if defs is None:
        defs = schema.get("$defs", {})
    errors: list[str] = []

    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        ref_schema = defs.get(ref_name, {})
        errors.extend(_check_openai_strict_compliance(ref_schema, f"{path}.$ref({ref_name})", defs))
        return errors

    if schema.get("type") != "object":
        # Check items in arrays
        if schema.get("type") == "array" and "items" in schema:
            errors.extend(_check_openai_strict_compliance(
                schema["items"], f"{path}.items", defs
            ))
        # Check anyOf branches
        if "anyOf" in schema:
            for i, branch in enumerate(schema["anyOf"]):
                errors.extend(_check_openai_strict_compliance(
                    branch, f"{path}.anyOf[{i}]", defs
                ))
        return errors

    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Rule 1: additionalProperties must be false
    if schema.get("additionalProperties") is not False:
        errors.append(f"{path}: additionalProperties is not false")

    # Rule 2: every property must be in required
    missing_required = set(props.keys()) - required
    if missing_required:
        errors.append(f"{path}: properties not in required: {sorted(missing_required)}")

    # Rule 3: no defaults allowed
    for prop_name, prop_schema in props.items():
        if "default" in prop_schema:
            errors.append(f"{path}.{prop_name}: has default value (not allowed in strict mode)")

    # Recurse into nested object/array properties
    for prop_name, prop_schema in props.items():
        errors.extend(_check_openai_strict_compliance(
            prop_schema, f"{path}.{prop_name}", defs
        ))

    return errors


def test_llm_normalization_schema_is_openai_strict_compatible() -> None:
    """The generated JSON schema must pass OpenAI strict-mode validation.

    This test catches the root cause: Pydantic fields with `= None` produce
    'default: null' and are omitted from 'required', which OpenAI rejects.
    """
    from loop_engine.models import LlmNormalizationBatch

    schema = LlmNormalizationBatch.model_json_schema()
    errors = _check_openai_strict_compliance(schema)

    assert errors == [], (
        "LLM schema is NOT OpenAI strict-mode compatible:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )

    # Also verify specific structural expectations
    candidate_schema = schema["$defs"]["LlmNormalizationCandidate"]
    candidate_props = set(candidate_schema["properties"].keys())
    candidate_required = set(candidate_schema["required"])
    assert candidate_props == candidate_required, (
        f"Properties/required mismatch: "
        f"extra in props={candidate_props - candidate_required}, "
        f"extra in required={candidate_required - candidate_props}"
    )
    assert candidate_schema["additionalProperties"] is False
    assert schema["additionalProperties"] is False
    assert "events" in set(schema.get("required", []))
