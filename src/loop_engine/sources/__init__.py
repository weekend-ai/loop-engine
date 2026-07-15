from __future__ import annotations

from loop_engine.config import AnalysisConfig, SourceConfig
from loop_engine.sources.base import EventSource
from loop_engine.sources.claude_jsonl import ClaudeCodeJsonlSource
from loop_engine.sources.claude_normalization import ClaudeSdkRecordNormalizer
from loop_engine.sources.litellm import LiteLLMLocalJsonSource, LiteLLMS3JsonSource


def build_source(
    config: SourceConfig, analysis: AnalysisConfig | None = None
) -> EventSource:
    if config.type == "claude_code_jsonl":
        if config.path is None:
            raise ValueError("Claude Code sources require path")
        normalizer = None
        if config.normalizer == "claude_sdk":
            if analysis is None:
                raise ValueError("Claude normalization requires analysis settings")
            normalizer = ClaudeSdkRecordNormalizer(
                model=analysis.model,
                timeout_seconds=analysis.timeout_seconds,
                max_input_chars=analysis.max_input_chars,
                max_record_chars=analysis.max_event_chars,
                max_output_tokens=analysis.max_output_tokens,
                redact_before_egress=analysis.redact_before_egress,
            )
        return ClaudeCodeJsonlSource(
            config.id,
            config.path,
            normalizer=normalizer,
            max_record_bytes=config.max_object_bytes,
            max_total_bytes=config.max_total_bytes,
        )
    if config.type == "litellm_local_json":
        if config.path is None:
            raise ValueError("local LiteLLM sources require path")
        return LiteLLMLocalJsonSource(config.id, config.path)
    if config.type == "litellm_s3_json":
        if config.uri is None:
            raise ValueError("S3 LiteLLM sources require uri")
        return LiteLLMS3JsonSource(
            config.id,
            config.uri,
            config.aws_profile,
            max_object_bytes=config.max_object_bytes,
            max_total_bytes=config.max_total_bytes,
        )
    raise ValueError(f"Unsupported source type: {config.type}")
