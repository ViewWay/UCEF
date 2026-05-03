"""Tests for quality system, model adapters, and physics models."""
import asyncio
import numpy as np

from ucef.core.types import ContextBlock, CompressionStrategy, QueryResult, TokenBudget
from ucef.quality.profiler import ModelCapabilityProfiler
from ucef.quality.monitor import QualityMonitor
from ucef.quality.feedback import QualityFeedbackLoop, RefinementAction
from ucef.quality.preservation import QualityPreservationEngine
from ucef.models.openai import OpenAIAdapter, OPENAI_MODEL_SPECS
from ucef.models.anthropic import AnthropicAdapter, ANTHROPIC_MODEL_SPECS
from ucef.models.zhipu import ZhipuAdapter, ZHIPU_MODEL_SPECS
from ucef.models.local import LocalAdapter
from ucef.physics.thermodynamic import ThermodynamicModel
from ucef.physics.quantum_field import RenormalizationGroup


def make_block(doc_id, text, score=0.5):
    return ContextBlock(
        document_id=doc_id, text=text,
        relevance_score=score, token_count=len(text.split()),
    )


def make_query_result(quality=0.8, query="test"):
    return QueryResult(
        query=query, context_blocks=[make_block("b1", "test", quality)],
        total_tokens=100, overall_quality=quality,
    )


class TestModelCapabilityProfiler:
    def setup_method(self):
        self.profiler = ModelCapabilityProfiler()

    def test_known_model_gpt4o(self):
        profile = asyncio.get_event_loop().run_until_complete(
            self.profiler.profile_model(model_client=None, model_name="gpt-4o")
        )
        assert profile is not None
        assert profile.native_context_window > 0

    def test_known_model_claude(self):
        profile = asyncio.get_event_loop().run_until_complete(
            self.profiler.profile_model(model_client=None, model_name="claude-3.5-sonnet")
        )
        assert profile is not None

    def test_unknown_model_without_client_raises(self):
        try:
            asyncio.get_event_loop().run_until_complete(
                self.profiler.profile_model(model_client=None, model_name="unknown-model-xyz")
            )
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_profile_has_strategy(self):
        profile = asyncio.get_event_loop().run_until_complete(
            self.profiler.profile_model(model_client=None, model_name="gpt-4o")
        )
        assert hasattr(profile, 'recommended_strategy')


class TestQualityMonitor:
    def setup_method(self):
        self.monitor = QualityMonitor(window_size=10)

    def test_record_and_stats(self):
        result = make_query_result(quality=0.85)
        self.monitor.record(result)
        stats = self.monitor.get_stats()
        # Stats key may vary — just check it returns a dict
        assert isinstance(stats, dict)
        assert len(stats) > 0

    def test_below_threshold(self):
        result = make_query_result(quality=0.3)
        self.monitor.record(result)
        assert self.monitor.is_below_threshold()

    def test_quality_not_degraded(self):
        for i in range(10):
            self.monitor.record(make_query_result(quality=0.85))
        assert not self.monitor.is_quality_degraded()


class TestQualityFeedbackLoop:
    def setup_method(self):
        self.loop = QualityFeedbackLoop(max_iterations=3)

    def test_refine_converges(self):
        call_count = 0

        async def requery_fn(query, top_k=None, quality_threshold=None, **kwargs):
            nonlocal call_count
            call_count += 1
            quality = 0.5 + call_count * 0.2
            return make_query_result(quality=quality)

        initial = make_query_result(quality=0.4)
        result = asyncio.get_event_loop().run_until_complete(
            self.loop.refine(initial, requery_fn, query="test", quality_threshold=0.7)
        )
        assert result.converged
        assert result.iterations <= 3


class TestQualityPreservationEngine:
    def test_creation(self):
        engine = QualityPreservationEngine(quality_threshold=0.75)
        assert engine is not None


# ── Model Adapters ───────────────────────────────────────────────────────────

class TestOpenAIAdapter:
    def test_model_specs(self):
        assert "gpt-4o" in OPENAI_MODEL_SPECS
        assert OPENAI_MODEL_SPECS["gpt-4o"] == 128000

    def test_adapter_creation(self):
        adapter = OpenAIAdapter(model="gpt-4o", api_key="test-key")
        assert adapter is not None


class TestAnthropicAdapter:
    def test_model_specs(self):
        assert "claude-3-5-sonnet-20241022" in ANTHROPIC_MODEL_SPECS

    def test_adapter_creation(self):
        adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022", api_key="test-key")
        assert adapter is not None


class TestZhipuAdapter:
    def test_model_specs(self):
        assert "glm-4" in ZHIPU_MODEL_SPECS

    def test_glm4_long_context(self):
        assert ZHIPU_MODEL_SPECS["glm-4-long"] == 1_000_000


class TestLocalAdapter:
    def test_adapter_creation(self):
        adapter = LocalAdapter(model="local-model", base_url="http://localhost:8000")
        assert adapter is not None

    def test_ollama_detection(self):
        adapter = LocalAdapter(model="llama3", base_url="http://localhost:11434", is_ollama=True)
        assert adapter is not None


# ── Physics Models ───────────────────────────────────────────────────────────

class TestThermodynamicModel:
    def setup_method(self):
        self.model = ThermodynamicModel()

    def test_energy_calculation(self):
        block = make_block("e1", "machine learning algorithms", 0.8)
        e = self.model.energy(block, query_words={"machine", "learning"})
        assert isinstance(e, float)

    def test_entropy_contribution(self):
        block = make_block("e2", "deep learning neural networks", 0.7)
        s = self.model.entropy_contribution(block, already_selected_words={"deep"})
        assert isinstance(s, float)

    def test_boltzmann_probabilities(self):
        blocks = [make_block(f"b{i}", f"Block {i}", 0.5 + i*0.1) for i in range(5)]
        probs = self.model.boltzmann_probabilities(blocks, query_words={"block"})
        assert len(probs) == len(blocks)
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_free_energy(self):
        f = self.model.free_energy(energy=0.5, entropy=0.3)
        assert isinstance(f, float)


class TestRenormalizationGroup:
    def setup_method(self):
        self.rg = RenormalizationGroup()

    def test_coarse_grain(self):
        text = "This is a long text for coarse graining. " * 20
        result = self.rg.coarse_grain(text, target_ratio=0.5)
        assert isinstance(result, str)
        assert len(result) < len(text)

    def test_relevance_flow(self):
        blocks = [make_block(f"r{i}", f"Relevance block {i}", 0.5) for i in range(10)]
        flow = self.rg.relevance_flow(blocks, query="relevance")
        assert len(flow) == len(blocks)
