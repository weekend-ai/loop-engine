from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable


class MessagesAPI(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class AnthropicClient(Protocol):
    @property
    def messages(self) -> MessagesAPI: ...


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def resolve_model(configured_model: str) -> str:
    return _first_env("LOOP_ENGINE_MODEL", "ANTHROPIC_MODEL", "LITELLM_MODEL") or configured_model


def build_anthropic_client(timeout_seconds: int) -> AnthropicClient:
    import anthropic

    api_key = _first_env("ANTHROPIC_API_KEY", "LITELLM_API_KEY", "LITELLM_MASTER_KEY")
    auth_token = _first_env("ANTHROPIC_AUTH_TOKEN")
    base_url = _first_env("ANTHROPIC_BASE_URL", "LITELLM_BASE_URL", "LITELLM_URL")
    if not api_key and not auth_token:
        raise RuntimeError(
            "Anthropic SDK authentication is not configured. Set ANTHROPIC_API_KEY "
            "(or LITELLM_API_KEY/LITELLM_MASTER_KEY), and set ANTHROPIC_BASE_URL "
            "for a LiteLLM proxy."
        )
    kwargs: dict[str, Any] = {
        "timeout": float(timeout_seconds),
        "max_retries": 2,
    }
    if auth_token:
        kwargs["auth_token"] = auth_token
    else:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)  # type: ignore[return-value]


def request_structured(
    client: AnthropicClient,
    *,
    model: str,
    system_prompt: str,
    payload: str,
    schema: dict[str, Any],
    max_output_tokens: int,
    operation: str,
) -> dict[str, Any]:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": payload}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            },
        )
    except Exception as error:
        raise RuntimeError(f"Anthropic SDK {operation} failed: {error}") from error

    text_parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)
    ]
    if not text_parts:
        raise RuntimeError(f"Anthropic SDK {operation} returned no text output")
    try:
        structured = json.loads("".join(text_parts))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Anthropic SDK {operation} returned invalid structured JSON"
        ) from error
    if not isinstance(structured, dict):
        raise RuntimeError(f"Anthropic SDK {operation} returned a non-object JSON value")
    return structured
