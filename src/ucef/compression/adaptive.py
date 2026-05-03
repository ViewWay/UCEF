"""
Adaptive Compressor — Strategy Router for Context Compression

Routes compression requests to the appropriate algorithm based on
CompressionStrategy, combining MDL, entropy, and task-aware methods.

Strategy mapping:
    - AGGRESSIVE: MDL hard truncation (retain 10%)
    - MODERATE:   Entropy + MDL (retain 30%)
    - LIGHT:      Task-aware extraction (retain 50%)
    - ADAPTIVE:   Auto-select based on quality_retention

References:
    - Grünwald, "The Minimum Description Length Principle", MIT Press 2007
    - Jaynes, "Information Theory and Statistical Mechanics", 1957
"""

from __future__ import annotations

from typing import Optional, Tuple

from ucef.core.config import CompressionConfig
from ucef.core.types import (
    CompressionResult,
    CompressionStrategy,
    ContextBlock,
    ModelClient,
    TokenBudget,
)

from ucef.compression.mdl import MDLCompressor
from ucef.compression.entropy import EntropyCompressor
from ucef.compression.task_aware import TaskAwareCompressor


class AdaptiveCompressor:
    """
    Adaptive context compressor with strategy routing.

    Selects and applies the optimal compression algorithm based on the
    configured strategy and model characteristics.
    """

    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        model_client: Optional[ModelClient] = None,
    ) -> None:
        self._config = config or CompressionConfig()
        self._mdl = MDLCompressor(
            description_length_weight=self._config.description_length_weight,
        )
        self._entropy = EntropyCompressor()
        self._task_aware = TaskAwareCompressor(model_client)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    async def compress(
        self,
        blocks: list,
        budget: TokenBudget,
        strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE,
        query: str = "",
        quality_retention: float = 0.8,
    ) -> Tuple[list, CompressionResult]:
        """
        Compress context blocks to fit within token budget.

        Args:
            blocks: List of ContextBlock to compress.
            budget: Token budget constraint.
            strategy: Compression strategy (or ADAPTIVE for auto-selection).
            query: User query for relevance-weighted compression.
            quality_retention: Model's quality retention capability (0-1).

        Returns:
            (compressed_blocks, compression_result)
        """
        if not blocks:
            return [], CompressionResult(original_tokens=0, compressed_tokens=0)

        # Resolve ADAPTIVE strategy
        resolved = self._resolve_strategy(strategy, quality_retention)

        max_tokens = budget.available_for_retrieval

        if resolved == CompressionStrategy.AGGRESSIVE:
            return self._compress_aggressive(blocks, max_tokens, query)
        elif resolved == CompressionStrategy.MODERATE:
            return self._compress_moderate(blocks, max_tokens, query)
        elif resolved == CompressionStrategy.LIGHT:
            return self._compress_light(blocks, max_tokens, query)
        else:
            # Default to moderate
            return self._compress_moderate(blocks, max_tokens, query)

    # ──────────────────────────────────────────────────────────────────────
    # Strategy Implementations
    # ──────────────────────────────────────────────────────────────────────

    def _compress_aggressive(
        self,
        blocks: list,
        budget: int,
        query: str,
    ) -> Tuple[list, CompressionResult]:
        """
        Aggressive compression: MDL-based hard selection.

        Retains ~10% of original context. Best for small context models
        where every token counts.
        """
        total_tokens = sum(b.token_count for b in blocks)
        target = max(1, int(total_tokens * self._config.aggressive_ratio))
        # Ensure at least 1 token budget even with tiny inputs
        effective_budget = max(1, min(budget, target))

        if self._config.use_mdl:
            try:
                return self._mdl.compress_blocks(blocks, effective_budget, query)
            except Exception:
                return self._truncate_by_relevance(blocks, effective_budget)
        else:
            return self._truncate_by_relevance(blocks, effective_budget)

    def _compress_moderate(
        self,
        blocks: list,
        budget: int,
        query: str,
    ) -> Tuple[list, CompressionResult]:
        """
        Moderate compression: Entropy + MDL selection.

        Retains ~30%. Balances information density with diversity.
        """
        target = max(1, int(sum(b.token_count for b in blocks) * self._config.moderate_ratio))
        effective_budget = max(1, min(budget, target))

        # First pass: entropy-based diversity selection
        if self._config.use_entropy:
            diverse, _ = self._entropy.compress_blocks(blocks, effective_budget, query)
            if diverse:
                # Second pass: MDL ranking within diverse set
                if self._config.use_mdl:
                    return self._mdl.compress_blocks(diverse, effective_budget, query)
                return diverse, CompressionResult(
                    original_tokens=sum(b.token_count for b in blocks),
                    compressed_tokens=sum(b.token_count for b in diverse),
                )

        return self._truncate_by_relevance(blocks, effective_budget)

    def _compress_light(
        self,
        blocks: list,
        budget: int,
        query: str,
    ) -> Tuple[list, CompressionResult]:
        """
        Light compression: Task-aware sentence extraction.

        Retains ~50%. Extracts key sentences from each block.
        """
        return self._task_aware.compress_blocks(blocks, budget, query)

    # ──────────────────────────────────────────────────────────────────────
    # Strategy Resolution
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_strategy(
        self,
        strategy: CompressionStrategy,
        quality_retention: float,
    ) -> CompressionStrategy:
        """
        Resolve ADAPTIVE strategy based on quality retention.

        High quality retention (>0.9) → can afford light compression
        Medium (0.7-0.9) → moderate
        Low (<0.7) → aggressive (maximize signal-to-noise)
        """
        if strategy != CompressionStrategy.ADAPTIVE:
            return strategy

        if quality_retention >= 0.9:
            return CompressionStrategy.LIGHT
        elif quality_retention >= 0.7:
            return CompressionStrategy.MODERATE
        else:
            return CompressionStrategy.AGGRESSIVE

    # ──────────────────────────────────────────────────────────────────────
    # Fallback
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _truncate_by_relevance(
        blocks: list,
        budget: int,
    ) -> Tuple[list, CompressionResult]:
        """Simple truncation fallback — keep highest-relevance blocks."""
        sorted_blocks = sorted(blocks, key=lambda b: b.relevance_score, reverse=True)
        original = sum(b.token_count for b in blocks)

        selected: list = []
        total = 0
        for block in sorted_blocks:
            if total + block.token_count <= budget:
                selected.append(block)
                total += block.token_count

        return selected, CompressionResult(
            original_tokens=original,
            compressed_tokens=total,
        )
