"""End-to-end integration test for the full UCEF pipeline."""
import asyncio
import numpy as np

from ucef import UniversalContextSystem, UCEFConfig, Document


class MockModelClient:
    """Simple mock model client for testing."""
    def generate(self, prompt: str, **kwargs) -> str:
        return f"Response to: {prompt[:50]}..."


class TestUniversalContextSystemE2E:
    def setup_method(self):
        config = UCEFConfig()
        config.hyperbolic.embedding_dim = 8
        config.quality.quality_threshold = 0.5
        self.system = UniversalContextSystem(
            model_client=MockModelClient(),
            model_name="gpt-4o",
            config=config,
        )
        asyncio.get_event_loop().run_until_complete(self.system.initialize())

    def test_store_documents_and_query(self):
        docs = [
            Document(id=f"doc{i}", text=f"Document about topic {i}. " * 10)
            for i in range(20)
        ]
        asyncio.get_event_loop().run_until_complete(self.system.store_documents(docs))
        result = asyncio.get_event_loop().run_until_complete(
            self.system.query("Tell me about topic 5")
        )
        assert result is not None

    def test_store_text_shortcut(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.system.store_text("Short text about machine learning", doc_id="ml1"))
        loop.run_until_complete(self.system.store_text("Another text about deep learning", doc_id="dl1"))
        result = loop.run_until_complete(self.system.query("machine learning"))
        assert result is not None

    def test_multiple_queries(self):
        loop = asyncio.get_event_loop()
        for i in range(5):
            loop.run_until_complete(self.system.store_text(f"Content about subject {i}", doc_id=f"s{i}"))
        for i in range(3):
            result = loop.run_until_complete(self.system.query(f"subject {i}"))
            assert result is not None

    def test_stats(self):
        loop = asyncio.get_event_loop()
        for i in range(5):
            loop.run_until_complete(self.system.store_text(f"Doc {i}", doc_id=f"d{i}"))
        loop.run_until_complete(self.system.query("test query"))
        stats = self.system.get_stats()
        assert stats is not None

    def test_quality_stats(self):
        loop = asyncio.get_event_loop()
        for i in range(5):
            loop.run_until_complete(self.system.store_text(f"Quality test doc {i}", doc_id=f"q{i}"))
        loop.run_until_complete(self.system.query("quality test"))
        quality_stats = self.system.get_quality_stats()
        assert quality_stats is not None


class TestSystemWithDifferentModels:
    def _make_system(self, model_name):
        config = UCEFConfig()
        config.hyperbolic.embedding_dim = 8
        system = UniversalContextSystem(
            model_client=MockModelClient(),
            model_name=model_name,
            config=config,
        )
        asyncio.get_event_loop().run_until_complete(system.initialize())
        return system

    def test_with_small_context_model(self):
        system = self._make_system("gpt-3.5-turbo")
        loop = asyncio.get_event_loop()
        for i in range(10):
            loop.run_until_complete(system.store_text(f"Small context doc {i}", doc_id=f"sc{i}"))
        result = loop.run_until_complete(system.query("test"))
        assert result is not None

    def test_with_claude(self):
        system = self._make_system("claude-3-5-sonnet-20241022")
        loop = asyncio.get_event_loop()
        for i in range(10):
            loop.run_until_complete(system.store_text(f"Claude context doc {i}", doc_id=f"cl{i}"))
        result = loop.run_until_complete(system.query("claude test"))
        assert result is not None

    def test_with_glm4(self):
        system = self._make_system("glm-4")
        loop = asyncio.get_event_loop()
        for i in range(10):
            loop.run_until_complete(system.store_text(f"GLM context doc {i}", doc_id=f"glm{i}"))
        result = loop.run_until_complete(system.query("glm test"))
        assert result is not None
