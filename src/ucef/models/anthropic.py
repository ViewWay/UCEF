"""
Anthropic Model Adapter — Claude 3.5 Sonnet, Claude 3 Opus, etc.

Supports Anthropic's Messages API with:
- Token counting via API estimation
- Retry with exponential backoff
- System prompt handling

References:
    - Anthropic API: https://docs.anthropic.com/en/docs/api-reference/messages
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ucef.models.base import BaseModelAdapter, AdapterConfig

logger = logging.getLogger(__name__)

# Known model specs: context window sizes
ANTHROPIC_MODEL_SPECS: Dict[str, int] = {
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-sonnet-latest": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
}


class AnthropicAdapter(BaseModelAdapter):
    """
    Anthropic API adapter for Claude models.

    Requires the `anthropic` package. Install with:
        pip install anthropic

    Usage:
        adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022", api_key="sk-ant-...")
        response = await adapter.generate("Explain quantum computing.")
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
    ) -> None:
        cfg = config or AdapterConfig()
        cfg.model_name = model
        super().__init__(cfg)

        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

        # Resolve context window
        self._context_window = ANTHROPIC_MODEL_SPECS.get(model, 200_000)

    def _ensure_client(self) -> Any:
        """Lazy-initialize the Anthropic client."""
        if self._client is not None:
            return self._client

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicAdapter. "
                "Install with: pip install anthropic"
            )

        kwargs: Dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url

        self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    async def _generate_impl(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Generate using Anthropic Messages API."""
        client = self._ensure_client()

        # Anthropic requires max_tokens
        api_max_tokens = min(max_tokens, 8192)

        response = await client.messages.create(
            model=self._model,
            max_tokens=api_max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        # Extract text from response
        content_blocks = response.content
        text_parts = [
            block.text for block in content_blocks
            if hasattr(block, "text")
        ]
        return "".join(text_parts)

    async def _count_tokens_impl(self, text: str) -> int:
        """
        Count tokens using Anthropic's token counting.

        Falls back to rough estimation if API is unavailable.
        """
        try:
            client = self._ensure_client()
            # Use the count_tokens beta endpoint if available
            result = await client.count_tokens(
                model=self._model,
                messages=[{"role": "user", "content": text}],
            )
            return result.input_tokens
        except Exception:
            # Rough estimation: ~3.5 chars per token for Claude models
            return max(1, len(text) // 4)
