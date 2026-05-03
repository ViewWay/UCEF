"""
Tests for Model Adapters — Base, Local, and adapter factory.

Validates token counting, retry logic, timeout handling, and adapter config.
"""

import asyncio
import pytest

from ucef.models.base import BaseModelAdapter, AdapterConfig, GenerationStats
from ucef.models.local import LocalAdapter


# ──────────────────────────────────────────────────────────────────────
# Concrete test adapter (avoids needing real API keys)
# ──────────────────────────────────────────────────────────────────────

class MockAdapter(BaseModelAdapter):
    """Test adapter that returns predictable responses."""

    def __init__(self, config=None, responses=None, fail_count=0):
        super().__init__(config)
        self._responses = responses or ["mock response"]
        self._call_count = 0
        self._fail_count = fail_count  # Fail this many times before succeeding

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def context_window(self) -> int:
        return 4096

    async def _generate_impl(self, prompt, max_tokens, temperature, **kwargs):
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise ConnectionError("Simulated API failure")
        return self._responses[(self._call_count - 1) % len(self._responses)]

    async def _count_tokens_impl(self, text: str) -> int:
        return max(1, len(text) // 4)


# ──────────────────────────────────────────────────────────────────────
# AdapterConfig Tests
# ──────────────────────────────────────────────────────────────────────

class TestAdapterConfig:
    def test_defaults(self):
        config = AdapterConfig()
        assert config.max_retries == 3
        assert config.retry_delay_base == 1.0
        assert config.timeout == 60.0
        assert config.default_temperature == 0.7
        assert config.context_window == 4096

    def test_custom_values(self):
        config = AdapterConfig(
            model_name="test-model",
            context_window=128000,
            max_retries=5,
            timeout=120.0,
        )
        assert config.model_name == "test-model"
        assert config.context_window == 128000
        assert config.max_retries == 5
        assert config.timeout == 120.0


# ──────────────────────────────────────────────────────────────────────
# BaseModelAdapter Tests
# ──────────────────────────────────────────────────────────────────────

class TestBaseAdapter:
    @pytest.mark.asyncio
    async def test_basic_generation(self):
        adapter = MockAdapter(responses=["Hello world"])
        result = await adapter.generate("test prompt")
        assert result == "Hello world"
        assert adapter._call_count == 1

    @pytest.mark.asyncio
    async def test_token_counting(self):
        adapter = MockAdapter()
        count = await adapter.count_tokens("This is a test sentence.")
        assert count > 0
        assert count >= len("This is a test sentence.") // 4 - 1

    @pytest.mark.asyncio
    async def test_token_counting_fallback_on_empty(self):
        adapter = MockAdapter()
        # Empty string should still return >= 1
        count = await adapter.count_tokens("")
        assert count >= 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        # Fail twice, then succeed
        adapter = MockAdapter(
            config=AdapterConfig(max_retries=3, retry_delay_base=0.01),
            fail_count=2,
        )
        result = await adapter.generate("test prompt")
        assert result == "mock response"
        assert adapter._call_count == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        adapter = MockAdapter(
            config=AdapterConfig(max_retries=2, retry_delay_base=0.01),
            fail_count=10,  # Always fail
        )
        with pytest.raises(ConnectionError):
            await adapter.generate("test prompt")

    @pytest.mark.asyncio
    async def test_generation_stats_tracking(self):
        adapter = MockAdapter(responses=["response 1", "response 2"])
        await adapter.generate("prompt 1")
        await adapter.generate("prompt 2")

        stats = adapter.get_stats()
        assert stats["total_calls"] == 2
        assert "mean_latency_ms" in stats
        assert stats["mean_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_stats_empty_initially(self):
        adapter = MockAdapter()
        stats = adapter.get_stats()
        assert stats["total_calls"] == 0


# ──────────────────────────────────────────────────────────────────────
# LocalAdapter Tests
# ──────────────────────────────────────────────────────────────────────

class TestLocalAdapter:
    def test_creation_default(self):
        adapter = LocalAdapter()
        assert adapter.model_name == "local"
        assert adapter.context_window > 0

    def test_creation_with_config(self):
        config = AdapterConfig(
            model_name="my-local-model",
            context_window=8192,
        )
        adapter = LocalAdapter(config=config)
        assert adapter.context_window == 8192

    @pytest.mark.asyncio
    async def test_generate_returns_string(self):
        adapter = LocalAdapter()
        result = await adapter.generate("Hello")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_token_counting(self):
        adapter = LocalAdapter()
        count = await adapter.count_tokens("Hello world, this is a test.")
        assert count > 0


# ──────────────────────────────────────────────────────────────────────
# GenerationStats Tests
# ──────────────────────────────────────────────────────────────────────

class TestGenerationStats:
    def test_defaults(self):
        stats = GenerationStats()
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0
        assert stats.total_tokens == 0
        assert stats.latency_ms == 0.0
        assert stats.finish_reason == ""

    def test_with_values(self):
        stats = GenerationStats(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=234.5,
            finish_reason="stop",
        )
        assert stats.prompt_tokens == 100
        assert stats.total_tokens == 150
        assert stats.finish_reason == "stop"
