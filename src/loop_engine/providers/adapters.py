from __future__ import annotations

from typing import Any

from loop_engine.providers.base import ProviderResponse


class AnthropicAdapter:
    """Anthropic Messages API with output_config JSON schema."""

    def __init__(
        self,
        timeout_seconds: int = 120,
        *,
        client: Any | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            from loop_engine.providers.registry import _first_env

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
                "timeout": float(self.timeout_seconds),
                "max_retries": 2,
            }
            if auth_token:
                kwargs["auth_token"] = auth_token
            else:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

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
        client = self._get_client()
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
            raise RuntimeError(f"Anthropic {operation} failed: {error}") from error

        text_parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
            and isinstance(getattr(block, "text", None), str)
        ]
        if not text_parts:
            raise RuntimeError(f"Anthropic {operation} returned no text output")
        return ProviderResponse(raw_text="".join(text_parts))


class OpenAIAdapter:
    """OpenAI Chat Completions API with response_format JSON schema.

    Compatible with any OpenAI-compatible API (OpenAI, Azure, vLLM, etc.).
    """

    def __init__(
        self,
        timeout_seconds: int = 120,
        *,
        client: Any | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            from loop_engine.providers.registry import _first_env

            api_key = _first_env("OPENAI_API_KEY", "LITELLM_API_KEY")
            base_url = _first_env("OPENAI_BASE_URL", "LITELLM_BASE_URL")
            if not api_key:
                raise RuntimeError(
                    "OpenAI API authentication is not configured. Set OPENAI_API_KEY "
                    "(or LITELLM_API_KEY)."
                )
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(self.timeout_seconds),
                max_retries=2,
            )
        return self._client

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
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_output_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": payload},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": operation.replace(" ", "_"),
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        except Exception as error:
            raise RuntimeError(f"OpenAI {operation} failed: {error}") from error

        choice = response.choices[0] if response.choices else None
        if choice is None or not choice.message or not choice.message.content:
            raise RuntimeError(f"OpenAI {operation} returned no content")
        return ProviderResponse(raw_text=choice.message.content)
