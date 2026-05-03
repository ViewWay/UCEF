"""
MDL Compressor — Minimum Description Length Principle for Context Compression

Implements context compression guided by the MDL principle:
    MDL = L(context) + L(query | context)
    minimize subject to: L(context) ≤ token_budget

The compressor evaluates each context block's "description cost" relative to
the query and selects blocks that minimize total description length within
the token budget.

References:
    - Grünwald, "The Minimum Description Length Principle", MIT Press 2007
    - Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference",
      EMNLP 2023
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from ucef.core.types import CompressionResult, CompressionStrategy, ContextBlock


class MDLCompressor:
    """
    MDL-based context compressor.

    For each block, computes:
        - L(block): self-information of the block's word distribution
        - L(query | block): cross-entropy between query and block distributions
    Then selects blocks that minimize total MDL within the budget.

    Complexity: O(n * m) where n = blocks, m = average word count.
    """

    def __init__(
        self,
        strategy: CompressionStrategy = CompressionStrategy.MODERATE,
        description_length_weight: float = 0.5,
    ) -> None:
        self._strategy = strategy
        self._dl_weight = description_length_weight

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
        Compress blocks to fit within token budget using MDL selection.

        Args:
            blocks: Candidate context blocks.
            budget: Maximum total tokens.
            query: User query for conditional description length.

        Returns:
            (selected_blocks, compression_result)
        """
        if not blocks:
            return [], CompressionResult(original_tokens=0, compressed_tokens=0)

        original_tokens = sum(b.token_count for b in blocks)

        # Compute MDL score for each block
        scored = self._score_blocks(blocks, query)

        # Sort by MDL score (lower is better — less description cost)
        scored.sort(key=lambda x: x[1])

        # Greedy selection within budget
        selected: List[ContextBlock] = []
        total_tokens = 0
        for block, _score in scored:
            if total_tokens + block.token_count <= budget:
                selected.append(block)
                total_tokens += block.token_count

        result = CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=total_tokens,
        )

        return selected, result

    def compress_block_text(
        self,
        text: str,
        target_ratio: float,
        query: str = "",
    ) -> str:
        """
        Compress a single block's text to approximately target_ratio of original.

        Sentence-level selection: split into sentences, score by MDL relevance
        to query, keep top sentences.

        Args:
            text: Block text to compress.
            target_ratio: Target retention ratio (0.0 - 1.0).
            query: User query for relevance scoring.

        Returns:
            Compressed text.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text

        n_keep = max(1, int(len(sentences) * target_ratio))

        # Score each sentence by query relevance
        query_words = set(query.lower().split()) if query else set()
        scored: List[Tuple[str, float]] = []

        for i, sent in enumerate(sentences):
            sent_words = set(sent.lower().split())
            # Relevance: word overlap with query
            overlap = len(query_words & sent_words) / max(len(query_words), 1) if query_words else 0.0
            # Position bonus: earlier sentences tend to be more important
            position_bonus = 1.0 - (i / len(sentences)) * 0.3
            # Self-information: prefer information-dense sentences
            info = self._self_information(sent)
            score = 0.4 * overlap + 0.3 * position_bonus + 0.3 * info
            scored.append((sent, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        kept = [s for s, _ in scored[:n_keep]]

        # Re-sort by original order
        kept_set = set(kept)
        result = [s for s in sentences if s in kept_set]
        return " ".join(result)

    # ──────────────────────────────────────────────────────────────────────
    # MDL Scoring
    # ──────────────────────────────────────────────────────────────────────

    def _score_blocks(
        self,
        blocks: List[ContextBlock],
        query: str,
    ) -> List[Tuple[ContextBlock, float]]:
        """Compute MDL score for each block."""
        # Build global word frequency for L(block) computation
        all_words: List[str] = []
        for block in blocks:
            all_words.extend(block.text.lower().split())
        global_freq = Counter(all_words)
        total_words = len(all_words) or 1

        query_words = set(query.lower().split()) if query else set()

        scored = []
        for block in blocks:
            # L(block): description length = -Σ log₂ P(word) / n_words
            l_block = self._description_length(block.text, global_freq, total_words)

            # L(query | block): conditional description length
            l_query_given = self._conditional_description_length(
                block.text, query_words
            )

            # MDL = w * L(block) + (1-w) * L(query | block)
            mdl = self._dl_weight * l_block + (1.0 - self._dl_weight) * l_query_given

            # Normalize by token count (prefer compact information)
            mdl_per_token = mdl / max(block.token_count, 1)

            scored.append((block, mdl_per_token))

        return scored

    def _description_length(
        self,
        text: str,
        global_freq: Counter,
        total_words: int,
    ) -> float:
        """
        Compute description length L(text) in bits.

        L(text) = -Σ_{w ∈ text} log₂ P(w)
        where P(w) = count(w) / total_words
        """
        words = text.lower().split()
        if not words:
            return 0.0

        bits = 0.0
        for w in words:
            count = global_freq.get(w, 1)
            prob = count / total_words
            bits -= math.log2(max(prob, 1e-10))

        return bits / len(words)  # Normalize per word

    def _conditional_description_length(
        self,
        block_text: str,
        query_words: set,
    ) -> float:
        """
        Compute L(query | block) — cost of describing query given block.

        Low cost means block is relevant to query (query words appear in block).
        High cost means block doesn't help explain the query.
        """
        if not query_words:
            return 1.0  # Neutral score when no query

        block_words = set(block_text.lower().split())
        overlap = len(query_words & block_words)
        coverage = overlap / len(query_words)

        # Inverse coverage as cost: more coverage = lower cost
        if coverage > 0:
            return -math.log2(max(coverage, 1e-10))
        else:
            return math.log2(len(query_words) + 1)  # Penalty for no overlap

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _self_information(text: str) -> float:
        """Compute average self-information of text (bits per word)."""
        words = text.lower().split()
        if not words:
            return 0.0
        freq = Counter(words)
        n = len(words)
        info = 0.0
        for count in freq.values():
            p = count / n
            info -= p * math.log2(max(p, 1e-10))
        return info

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = []
        for s in text.replace("\n", ". ").split("."):
            s = s.strip()
            if s:
                sentences.append(s)
        return sentences if sentences else [text]
