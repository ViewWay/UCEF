"""
Zhipu AI Model Adapter — GLM-4, ChatGLM series

Supports Zhipu AI's API (compatible with OpenAI format) with:
- Token counting via API estimation
- Retry with exponential backoff
- SSE streaming support (TODO)

References:
    - Zhipu API: https://open.bigmodel.cn/dev/api
    - zhipuai SDK: https://github.com/zhipuai/zhipuai-sdk-python-v4
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ucef.models.base import BaseModelAdapter, AdapterConfig

logger = logging.getLogger(__name__)

# Known model specs: context window sizes
ZHIPU_MODEL_SPECS: Dict[str, int] = {
    "glm-4-plus": 128_000,
    "glm-4": 128_000,
    "glm-4-long": 1_000_000,
    "glm-4-flash": 128_000,
    "glm-4-flashx": 128_000,
    "glm-4-air": 128_000,
    "glm-4-airx": 128_000,
    "glm-4v": 8_192,
    "glm-4v-plus": 8_192,
    "chatglm-turbo": 32_768,
    "chatglm_pro": 32_768,
    "chatglm_std": 32_768,
    "chatglm_lite": 32_768,
}


class ZhipuAdapter(BaseModelAdapter):
    """
    Zhipu AI API adapter for GLM models.

    Requires the `zhipuai` package. Install with:
        pip install zhipuai

    Usage:
        adapter = ZhipuAdapter(model="glm-4", api_key="your-key")
        response = await adapter.generate("解释量子计算。")
    """

    def __init__(
        self,
        model: str = "glm-4",
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
        self._context_window = ZHIPU_MODEL_SPECS.get(model, 32_768)

    def _ensure_client(self) -> Any:
        """Lazy-initialize the Zhipu AI client."""
        if self._client is not None:
            return self._client

        try:
            from zhipuai import ZhipuAI
        except ImportError:
            raise ImportError(
                "zhipuai package is required for ZhipuAdapter. "
                "Install with: pip install zhipuai"
            )

        kwargs: Dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url

        self._client = ZhipuAI(**kwargs)
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
        """
        Generate using Zhipu AI API (OpenAI-compatible format).

        The zhipuai SDK is synchronous, so we wrap in asyncio.
        """
        client = self._ensure_client()

        import asyncio

        def _sync_call() -> str:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            content = response.choices[0].message.content
            return content or ""

        return await asyncio.get_running_loop().run_in_executor(None, _sync_call)

    async def _count_tokens_impl(self, text: str) -> int:
        """
        Count tokens for GLM models.

        Uses rough estimation. For precise counting, would need
        the ChatGLM tokenizer (sentencepiece-based).
        """
        # GLM models use ~2 chars per token for Chinese, ~4 for English
        # Mixed content heuristic: weight by CJK character ratio
        cjk_count = sum(1 for c in text if '一' <= c <= '鿿')
        ratio = cjk_count / max(len(text), 1)

        # Interpolate between 2 chars/token (Chinese) and 4 chars/token (English)
        chars_per_token = 2.0 * ratio + 4.0 * (1.0 - ratio)
        return max(1, int(len(text) / chars_per_token))
