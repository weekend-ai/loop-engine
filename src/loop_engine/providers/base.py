from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ProviderResponse:
    """Normalized response from any structured-output provider."""

    raw_text: str
    """The raw JSON text returned by the provider."""


class ProviderAdapter(Protocol):
    """Protocol for structured-output provider adapters.

    Each adapter translates the same closed JSON Schema into the
    provider's native request format (Anthropic output_config,
    OpenAI response_format, etc.) and normalizes the response.
    """

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
        """Send a structured-output request and return the raw JSON text."""
        ...
