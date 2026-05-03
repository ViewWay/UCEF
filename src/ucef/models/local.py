"""
Local Model Adapter — llama.cpp, vLLM, Ollama, and other local inference engines

Supports local model inference via:
- llama.cpp server (OpenAI-compatible API)
- vLLM server (OpenAI-compatible API)
- Ollama (native API)
- Any OpenAI-compatible local server

All use the OpenAI Chat Completions format over HTTP.

References:
    - llama.cpp: https://github.com/ggerganov/llama.cpp
    - vLLM: https://github.com/vllm-project/vllm
    - Ollama: https://github.com/ollama/ollama
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from ucef.models.base import BaseModelAdapter, AdapterConfig

logger = logging.getLogger(__name__)


class LocalAdapter(BaseModelAdapter):
    """
    Local inference server adapter.

    Works with any OpenAI-compatible local server:
    - llama.cpp server: `./server -m model.gguf --port 8080`
    - vLLM: `python -m vllm.entrypoints.openai.api_server --model ...`
    - Ollama: `ollama serve` (runs on http://localhost:11434)

    Usage:
        adapter = LocalAdapter(
            model="llama-3-8b",
            base_url="http://localhost:8080",
            context_window=8192,
        )
        response = await adapter.generate("Explain quantum computing.")
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:8080",
        context_window: int = 4096,
        api_key: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
        is_ollama: bool = False,
    ) -> None:
        cfg = config or AdapterConfig()
        cfg.model_name = model
        cfg.base_url = base_url
        super().__init__(cfg)

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._context_window = context_window
        self._api_key = api_key
        self._is_ollama = is_ollama

        # Lazy-loaded HTTP session
        self._session = None

    async def _ensure_session(self) -> Any:
        """Lazy-initialize HTTP session."""
        if self._session is not None:
            return self._session

        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp package is required for LocalAdapter. "
                "Install with: pip install aiohttp"
            )

        self._session = aiohttp.ClientSession()
        return self._session

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
        """Generate using local server API."""
        session = await self._ensure_session()

        if self._is_ollama:
            return await self._generate_ollama(session, prompt, max_tokens, temperature)
        else:
            return await self._generate_openai_compat(session, prompt, max_tokens, temperature)

    async def _generate_openai_compat(
        self,
        session: Any,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using OpenAI-compatible API (llama.cpp, vLLM)."""
        url = f"{self._base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

    async def _generate_ollama(
        self,
        session: Any,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using Ollama API."""
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("message", {}).get("content", "")

    async def _count_tokens_impl(self, text: str) -> int:
        """
        Count tokens via local server.

        Tries the /v1/tokenize endpoint first (llama.cpp), falls back to estimation.
        """
        session = await self._ensure_session()

        # Try llama.cpp tokenize endpoint
        try:
            url = f"{self._base_url}/tokenize"
            payload = {"content": text}
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return len(data.get("tokens", []))
        except Exception:
            pass

        # Try Ollama tokenize
        if self._is_ollama:
            try:
                url = f"{self._base_url}/api/tokenize"
                payload = {"model": self._model, "prompt": text}
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return len(data.get("tokens", []))
            except Exception:
                pass

        # Fallback: rough estimation
        return max(1, len(text) // 4)

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> LocalAdapter:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
