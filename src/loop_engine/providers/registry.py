from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel

from loop_engine.providers.adapters import AnthropicAdapter, OpenAIAdapter
from loop_engine.providers.base import ProviderAdapter, ProviderResponse


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def resolve_model(configured_model: str) -> str:
    return _first_env("LOOP_ENGINE_MODEL", "ANTHROPIC_MODEL", "LITELLM_MODEL") or configured_model


def build_provider(
    provider: Literal["anthropic", "openai"] = "anthropic",
    timeout_seconds: int = 120,
    *,
    client: Any | None = None,
) -> ProviderAdapter:
    """Build a provider adapter from config.

    Returns an adapter that translates the same closed JSON schema
    into the provider's native request format.
    """
    adapter: ProviderAdapter
    if provider == "openai":
        adapter = OpenAIAdapter(timeout_seconds=timeout_seconds, client=client)
    else:
        adapter = AnthropicAdapter(timeout_seconds=timeout_seconds, client=client)
    return adapter


def parse_structured_response(
    response: ProviderResponse,
    target_type: type[BaseModel],
    operation: str,
) -> BaseModel:
    """Parse and validate a provider response against a Pydantic model.

    This is the single validation gate — all provider responses pass
    through here before entering the pipeline.
    """
    try:
        structured = json.loads(response.raw_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{operation} returned invalid structured JSON"
        ) from error
    if not isinstance(structured, dict):
        raise RuntimeError(f"{operation} returned a non-object JSON value")
    return target_type.model_validate(structured)


def request_and_validate(
    provider: ProviderAdapter,
    *,
    model: str,
    system_prompt: str,
    payload: str,
    target_type: type[BaseModel],
    max_output_tokens: int,
    operation: str,
    repair: bool = True,
) -> BaseModel:
    """Request structured output and validate. Optionally repair once on failure.

    If the provider returns invalid structured output and repair=True,
    sends the response, validation errors, and target schema back for
    one correction attempt. This is a bounded retry, not a permanent layer.
    """
    schema = target_type.model_json_schema()
    response = provider.request_structured(
        model=model,
        system_prompt=system_prompt,
        payload=payload,
        schema=schema,
        max_output_tokens=max_output_tokens,
        operation=operation,
    )
    try:
        return parse_structured_response(response, target_type, operation)
    except (RuntimeError, Exception) as first_error:
        if not repair:
            raise
        # Bounded repair: one retry with error feedback
        repair_prompt = (
            f"Your previous response failed validation.\n\n"
            f"Error: {first_error}\n\n"
            f"Your response was:\n{response.raw_text[:2000]}\n\n"
            f"Target JSON schema:\n{json.dumps(schema, indent=2)[:2000]}\n\n"
            f"Please return a corrected response matching the schema exactly."
        )
        try:
            repair_response = provider.request_structured(
                model=model,
                system_prompt=system_prompt,
                payload=repair_prompt,
                schema=schema,
                max_output_tokens=max_output_tokens,
                operation=f"{operation} (repair)",
            )
            return parse_structured_response(repair_response, target_type, f"{operation} (repair)")
        except Exception as repair_error:
            raise RuntimeError(
                f"{operation} failed validation and repair also failed: "
                f"original={first_error}, repair={repair_error}"
            ) from repair_error
