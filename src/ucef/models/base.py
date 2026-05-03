"""
Base Model Adapter — Common functionality for all model adapters.

Provides shared token counting, error handling, and retry logic
that concrete adapters (OpenAI, Anthropic, Zhipu, Local) build upon.

References:
    - UCEF Architecture: docs/api/architecture.md
    - ModelClient Protocol: ucef.core.types.ModelClient
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """Common configuration for model adapters."""
    model_name: str = ""
    context_window: int = 4096
    max_retries: int = 3
    retry_delay_base: float = 1.0  # seconds, exponential backoff base
    timeout: float = 60.0  # seconds
    default_temperature: float = 0.7
    default_max_tokens: int = 4096


@dataclass
class GenerationStats:
    """Statistics from a single generation call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    model_version: str = ""


class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.

    Provides retry logic, timeout handling, and common utilities.
    Subclasses implement _generate_impl() and _count_tokens_impl().
    """

    def __init__(self, config: Optional[AdapterConfig] = None) -> None:
        self._config = config or AdapterConfig()
        self._stats: List[GenerationStats] = []

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Return the native context window size in tokens."""
        ...

    @abstractmethod
    async def _generate_impl(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Subclass-specific generation logic (no retry/timeout wrapping)."""
        ...

    @abstractmethod
    async def _count_tokens_impl(self, text: str) -> int:
        """Subclass-specific token counting logic."""
        ...

    # ──────────────────────────────────────────────────────────────────────
    # Public API (implements ModelClient Protocol)
    # ──────────────────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response with retry logic and timeout.

        Wraps _generate_impl with exponential backoff on transient errors.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._config.max_retries):
            try:
                start = time.monotonic()
                result = await asyncio.wait_for(
                    self._generate_impl(prompt, max_tokens, temperature, **kwargs),
                    timeout=self._config.timeout,
                )
                elapsed = (time.monotonic() - start) * 1000

                self._stats.append(GenerationStats(
                    latency_ms=elapsed,
                    finish_reason="stop",
                    model_version=self.model_name,
                ))

                return result

            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout on attempt {attempt + 1}/{self._config.max_retries} "
                    f"for model {self.model_name}"
                )
                last_error = TimeoutError(
                    f"Generation timed out after {self._config.timeout}s"
                )

            except Exception as e:
                logger.warning(
                    f"Error on attempt {attempt + 1}/{self._config.max_retries}: {e}"
                )
                last_error = e

            # Exponential backoff
            if attempt < self._config.max_retries - 1:
                delay = self._config.retry_delay_base * (2 ** attempt)
                await asyncio.sleep(delay)

        # All retries exhausted
        raise last_error or RuntimeError("All retries exhausted")

    async def count_tokens(self, text: str) -> int:
        """Count tokens using the adapter's tokenizer."""
        try:
            return await self._count_tokens_impl(text)
        except Exception:
            # Fallback: rough estimation (~4 chars per token for English)
            return max(1, len(text) // 4)

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        if not self._stats:
            return {"total_calls": 0}

        latencies = [s.latency_ms for s in self._stats]
        return {
            "total_calls": len(self._stats),
            "mean_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
        }
