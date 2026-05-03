"""
OpenAI Model Adapter — GPT-4o, GPT-4-turbo, GPT-3.5-turbo

Supports OpenAI's Chat Completions API with:
- Token counting via tiktoken (when available) or API estimation
- Retry with exponential backoff
- Streaming support (TODO)

References:
    - OpenAI API: https://platform.openai.com/docs/api-reference/chat
    - tiktoken: https://github.com/openai/tiktoken
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ucef.models.base import BaseModelAdapter, AdapterConfig

logger = logging.getLogger(__name__)

# Known model specs: context window sizes
OPENAI_MODEL_SPECS: Dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4-turbo-preview": 128_000,
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-3.5-turbo": 16_385,
    "gpt-3.5-turbo-16k": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
}


class OpenAIAdapter(BaseModelAdapter):
    """
    OpenAI API adapter for GPT models.

    Requires the `openai` package. Install with:
        pip install openai

    Usage:
        adapter = OpenAIAdapter(model="gpt-4o", api_key="sk-...")
        response = await adapter.generate("Explain quantum computing.")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
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

        # Resolve context window from known specs
        self._context_window = OPENAI_MODEL_SPECS.get(model, 4_096)

        # Tokenizer (lazy loaded)
        self._tokenizer = None

    def _ensure_client(self) -> Any:
        """Lazy-initialize the OpenAI client."""
        if self._client is not None:
            return self._client

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIAdapter. "
                "Install with: pip install openai"
            )

        kwargs: Dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url

        self._client = AsyncOpenAI(**kwargs)
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
        """Generate using OpenAI Chat Completions API."""
        client = self._ensure_client()

        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        # Update stats with token counts from API response
        if response.usage and self._stats:
            self._stats[-1].prompt_tokens = response.usage.prompt_tokens
            self._stats[-1].completion_tokens = response.usage.completion_tokens
            self._stats[-1].total_tokens = response.usage.total_tokens

        content = response.choices[0].message.content
        return content or ""

    async def _count_tokens_impl(self, text: str) -> int:
        """Count tokens using tiktoken when available."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.encoding_for_model(self._model)
            except (ImportError, KeyError):
                # Fallback: use cl100k_base encoding or rough estimate
                try:
                    import tiktoken
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
                except ImportError:
                    self._tokenizer = None

        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))

        # Rough fallback
        return max(1, len(text) // 4)
