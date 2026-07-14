from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConfig(StrictConfigModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    type: Literal["claude_code_jsonl", "litellm_local_json", "litellm_s3_json"]
    path: str | None = None
    uri: str | None = None
    aws_profile: str | None = None
    max_object_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_total_bytes: int = Field(default=100 * 1024 * 1024, ge=1)

    @model_validator(mode="after")
    def validate_location(self) -> SourceConfig:
        if self.type == "litellm_s3_json" and not self.uri:
            raise ValueError("S3 sources require uri")
        if self.type != "litellm_s3_json" and not self.path:
            raise ValueError("local sources require path")
        if self.max_total_bytes < self.max_object_bytes:
            raise ValueError("max_total_bytes must be >= max_object_bytes")
        return self


class AnalysisConfig(StrictConfigModel):
    provider: Literal["rule_based", "claude_cli"] = "rule_based"
    model: str = "sonnet"
    max_concurrency: int = Field(default=4, ge=1)
    timeout_seconds: int = Field(default=120, ge=1, le=3600)
    max_input_chars: int = Field(default=100_000, ge=100)
    max_event_chars: int = Field(default=4_000, ge=100)
    external_data_egress_allowed: bool = False

    @model_validator(mode="after")
    def require_external_egress_opt_in(self) -> AnalysisConfig:
        if self.provider == "claude_cli" and not self.external_data_egress_allowed:
            raise ValueError(
                "claude_cli requires external_data_egress_allowed: true because task "
                "content is sent to the configured Claude provider"
            )
        return self


def _default_group_by() -> list[Literal["task_type", "model", "asset_version"]]:
    return ["task_type"]


class MetricsConfig(StrictConfigModel):
    group_by: list[Literal["task_type", "model", "asset_version"]] = Field(
        default_factory=_default_group_by
    )


def _default_formats() -> list[Literal["json", "markdown"]]:
    return ["json", "markdown"]


class OutputConfig(StrictConfigModel):
    directory: str = "./output"
    formats: list[Literal["json", "markdown"]] = Field(default_factory=_default_formats)


class EngineConfig(StrictConfigModel):
    version: Literal[1] = 1
    workspace: str = ".loop-engine"
    sources: list[SourceConfig]
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def validate_unique_source_ids(self) -> EngineConfig:
        source_ids = [source.id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source ids must be unique")
        return self


def load_config(path: Path) -> EngineConfig:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return EngineConfig.model_validate(raw)
