"""
Entropy Compressor — Maximum Entropy Principle for Context Selection

Selects context blocks that maximize information entropy while respecting
token budget constraints. Ensures diversity and minimizes redundancy.

    H(selected) = -Σ p_i log₂ p_i
    maximize H subject to Σ tokens_i ≤ budget

References:
    - Jaynes, "Information Theory and Statistical Mechanics", Physical Review 1957
    - Carbonell & Goldstein, "MMR: Maximal Marginal Relevance", SIGIR 1998
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ucef.core.types import CompressionResult, ContextBlock


class EntropyCompressor:
    """
    Entropy-maximizing context compressor.

    Greedy selection that maximizes marginal information entropy contribution
    at each step, penalizing redundancy with already-selected blocks.

    Complexity: O(n * k) where n = candidates, k = selected count.
    """

    def __init__(
        self,
        redundancy_threshold: float = 0.7,
        diversity_weight: float = 0.3,
    ) -> None:
        self._redundancy_threshold = redundancy_threshold
        self._diversity_weight = diversity_weight

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def compress_blocks(
        self,
        blocks: List[ContextBlock],
        budget: int,
        query: str = "",
    ) -> Tuple[List[ContextBlock], CompressionResult]:
        """
        Select diverse subset of blocks maximizing entropy within budget.

        Uses greedy marginal entropy: at each step, pick the block that
        maximizes relevance * (1 - redundancy_with_selected).

        Args:
            blocks: Candidate context blocks.
            budget: Maximum total tokens.
            query: User query for relevance weighting.

        Returns:
            (selected_blocks, compression_result)
        """
        if not blocks:
            return [], CompressionResult(original_tokens=0, compressed_tokens=0)

        original_tokens = sum(b.token_count for b in blocks)

        # Pre-compute word sets for redundancy calculation
        block_words = [set(b.text.lower().split()) for b in blocks]
        query_words = set(query.lower().split()) if query else set()

        selected: List[ContextBlock] = []
        selected_indices: Set[int] = set()
        total_tokens = 0

        for _ in range(len(blocks)):
            best_idx = -1
            best_score = -float("inf")

            for i, block in enumerate(blocks):
                if i in selected_indices:
                    continue
                if total_tokens + block.token_count > budget:
                    continue

                # Relevance component
                relevance = self._query_relevance(block_words[i], query_words)

                # Diversity component (marginal entropy)
                diversity = self._marginal_diversity(
                    block_words[i],
                    [block_words[j] for j in selected_indices],
                )

                # Combined score: relevance + λ * diversity
                score = (1.0 - self._diversity_weight) * relevance + \
                        self._diversity_weight * diversity

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1:
                break

            selected_indices.add(best_idx)
            selected.append(blocks[best_idx])
            total_tokens += blocks[best_idx].token_count

        result = CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=total_tokens,
        )

        return selected, result

    def block_entropy(self, block: ContextBlock) -> float:
        """
        Compute information entropy of a block's word distribution.

        H = -Σ p(w) log₂ p(w)
        """
        words = block.text.lower().split()
        if not words:
            return 0.0

        freq = Counter(words)
        n = len(words)
        entropy = 0.0
        for count in freq.values():
            p = count / n
            entropy -= p * math.log2(max(p, 1e-10))
        return entropy

    def redundancy_score(
        self,
        block_a: ContextBlock,
        block_b: ContextBlock,
    ) -> float:
        """
        Compute Jaccard similarity (redundancy) between two blocks.

        Returns:
            Value in [0, 1]. 0 = completely different, 1 = identical.
        """
        words_a = set(block_a.text.lower().split())
        words_b = set(block_b.text.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    # ──────────────────────────────────────────────────────────────────────
    # Internal Methods
    # ──────────────────────────────────────────────────────────────────────

    def _query_relevance(
        self,
        block_words: Set[str],
        query_words: Set[str],
    ) -> float:
        """Compute relevance of block to query."""
        if not query_words:
            return 0.5  # Neutral when no query

        overlap = len(block_words & query_words)
        return overlap / len(query_words)

    def _marginal_diversity(
        self,
        candidate_words: Set[str],
        selected_word_sets: List[Set[str]],
    ) -> float:
        """
        Compute marginal diversity contribution of candidate.

        Diversity = 1 - max_redundancy_with_selected
        (analogous to MMR: Maximal Marginal Relevance)
        """
        if not selected_word_sets:
            return 1.0  # First selection is maximally diverse

        max_redundancy = 0.0
        for sel_words in selected_word_sets:
            if not candidate_words or not sel_words:
                continue
            jaccard = len(candidate_words & sel_words) / len(candidate_words | sel_words)
            max_redundancy = max(max_redundancy, jaccard)

        return 1.0 - max_redundancy

    def _total_entropy(self, blocks: List[ContextBlock]) -> float:
        """
        Compute total entropy of a block collection.

        H_total = -Σ_{w ∈ all_words} P(w) log₂ P(w)
        where P(w) is computed across all blocks' combined text.
        """
        all_words: List[str] = []
        for block in blocks:
            all_words.extend(block.text.lower().split())

        if not all_words:
            return 0.0

        freq = Counter(all_words)
        n = len(all_words)
        entropy = 0.0
        for count in freq.values():
            p = count / n
            entropy -= p * math.log2(max(p, 1e-10))
        return entropy
