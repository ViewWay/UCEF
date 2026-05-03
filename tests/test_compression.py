"""Tests for compression strategies: MDL, entropy, task-aware, adaptive."""
import numpy as np

from ucef.core.types import ContextBlock, CompressionStrategy, TokenBudget
from ucef.compression.mdl import MDLCompressor
from ucef.compression.entropy import EntropyCompressor
from ucef.compression.task_aware import TaskAwareCompressor
from ucef.compression.adaptive import AdaptiveCompressor


def make_blocks(n=20, prefix="Block"):
    return [
        ContextBlock(
            document_id=f"blk{i}",
            text=f"{prefix} {i}: This is sentence one. This is sentence two. "
                 f"This is sentence three with more content about topic {i % 5}.",
            relevance_score=1.0 - (i * 0.04),
            token_count=20 + i,
        )
        for i in range(n)
    ]


class TestMDLCompressor:
    def setup_method(self):
        self.compressor = MDLCompressor()

    def test_compress_blocks(self):
        blocks = make_blocks(20)
        result, info = self.compressor.compress_blocks(blocks, budget=200)
        assert len(result) > 0
        assert len(result) <= len(blocks)

    def test_compress_preserves_ids(self):
        blocks = make_blocks(5)
        result, info = self.compressor.compress_blocks(blocks, budget=500)
        result_ids = {b.document_id for b in result}
        original_ids = {b.document_id for b in blocks}
        assert result_ids.issubset(original_ids)

    def test_compress_empty_input(self):
        result, info = self.compressor.compress_blocks([], budget=100)
        assert result == []

    def test_compress_with_query(self):
        blocks = make_blocks(10)
        result, info = self.compressor.compress_blocks(blocks, budget=200, query="topic 3")
        assert len(result) > 0


class TestEntropyCompressor:
    def setup_method(self):
        self.compressor = EntropyCompressor()

    def test_compress_blocks(self):
        blocks = make_blocks(20)
        result, info = self.compressor.compress_blocks(blocks, budget=300)
        assert len(result) > 0

    def test_compress_empty(self):
        result, info = self.compressor.compress_blocks([], budget=100)
        assert result == []


class TestTaskAwareCompressor:
    def setup_method(self):
        self.compressor = TaskAwareCompressor()

    def test_compress_blocks(self):
        blocks = make_blocks(20)
        result, info = self.compressor.compress_blocks(blocks, budget=300, query="topic 2")
        assert len(result) > 0

    def test_compress_without_query(self):
        blocks = make_blocks(10)
        result, info = self.compressor.compress_blocks(blocks, budget=200)
        assert len(result) > 0


class TestAdaptiveCompressor:
    def setup_method(self):
        self.compressor = AdaptiveCompressor()

    def test_aggressive_strategy(self):
        import asyncio
        blocks = make_blocks(30)
        budget = TokenBudget(total=100)
        result, info = asyncio.get_event_loop().run_until_complete(
            self.compressor.compress(
                blocks, budget=budget,
                strategy=CompressionStrategy.AGGRESSIVE, query="topic 1"
            )
        )
        assert len(result) > 0

    def test_light_strategy(self):
        import asyncio
        blocks = make_blocks(10)
        budget = TokenBudget(total=500)
        result, info = asyncio.get_event_loop().run_until_complete(
            self.compressor.compress(
                blocks, budget=budget,
                strategy=CompressionStrategy.LIGHT, query="topic 3"
            )
        )
        assert len(result) > 0

    def test_empty_input(self):
        import asyncio
        budget = TokenBudget(total=100)
        result, info = asyncio.get_event_loop().run_until_complete(
            self.compressor.compress(
                [], budget=budget,
                strategy=CompressionStrategy.AGGRESSIVE
            )
        )
        assert result == []
