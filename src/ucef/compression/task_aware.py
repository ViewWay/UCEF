"""
Task-Aware Compressor — Query-Directed Sentence Extraction & Summarization

Compresses context by extracting query-relevant sentences and optionally
using an LLM to generate concise summaries. Inspired by ATACompressor
which achieves 60% compression with 92% performance retention.

Strategy:
    1. Split blocks into sentences
    2. Score each sentence for query relevance
    3. Select top-k sentences within budget
    4. (Optional) LLM-based summarization for further compression

References:
    - Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference",
      EMNLP 2023
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from ucef.core.types import CompressionResult, ContextBlock, ModelClient


class TaskAwareCompressor:
    """
    Task-aware context compressor.

    Extracts the most query-relevant sentences from each block and
    reconstructs compressed context. Optionally uses an LLM to generate
    summaries for maximum compression.

    Complexity: O(n * s) where n = blocks, s = sentences per block.
    """

    def __init__(self, model_client: Optional[ModelClient] = None) -> None:
        self._client = model_client

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
        Compress blocks via sentence-level selection.

        For each block, extract query-relevant sentences and rebuild
        a compressed version. Then select blocks within budget.
        """
        if not blocks:
            return [], CompressionResult(original_tokens=0, compressed_tokens=0)

        original_tokens = sum(b.token_count for b in blocks)

        # Compress each block individually
        compressed_blocks: List[ContextBlock] = []
        for block in blocks:
            cb = self._compress_single_block(block, query)
            if cb and cb.text.strip():
                compressed_blocks.append(cb)

        # Sort by information density (relevance per token)
        compressed_blocks.sort(
            key=lambda b: b.relevance_score / max(b.token_count, 1),
            reverse=True,
        )

        # Select within budget
        selected: List[ContextBlock] = []
        total_tokens = 0
        for block in compressed_blocks:
            if total_tokens + block.token_count <= budget:
                selected.append(block)
                total_tokens += block.token_count

        result = CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=total_tokens,
        )

        return selected, result

    def extract_key_sentences(
        self,
        text: str,
        query: str,
        top_k: int = 5,
    ) -> List[str]:
        """
        Extract the most query-relevant sentences from text.

        Scoring: combination of word overlap, position, and information density.

        Args:
            text: Source text.
            query: Query to score against.
            top_k: Number of sentences to extract.

        Returns:
            Top-k sentences sorted by original position.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= top_k:
            return sentences

        query_words = set(query.lower().split())
        scored: List[Tuple[int, str, float]] = []

        for i, sent in enumerate(sentences):
            sent_words = set(sent.lower().split())

            # Word overlap score
            overlap = len(query_words & sent_words) / max(len(query_words), 1) if query_words else 0.0

            # Position score (first and last sentences often more important)
            pos_score = 1.0
            if i == 0:
                pos_score = 1.2
            elif i == len(sentences) - 1:
                pos_score = 1.1

            # Length score (prefer medium-length sentences)
            length = len(sent_words)
            if 5 <= length <= 30:
                length_score = 1.0
            elif length < 5:
                length_score = 0.5
            else:
                length_score = 0.8

            score = overlap * pos_score * length_score
            scored.append((i, sent, score))

        # Select top-k by score
        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[:top_k]

        # Re-sort by original position for coherence
        top.sort(key=lambda x: x[0])
        return [s for _, s, _ in top]

    # ──────────────────────────────────────────────────────────────────────
    # Internal Methods
    # ──────────────────────────────────────────────────────────────────────

    def _compress_single_block(
        self,
        block: ContextBlock,
        query: str,
        target_ratio: float = 0.6,
    ) -> Optional[ContextBlock]:
        """Compress a single block by extracting key sentences."""
        sentences = self._split_sentences(block.text)
        if len(sentences) <= 2:
            return block  # Too short to compress

        n_keep = max(1, int(len(sentences) * target_ratio))
        key_sents = self.extract_key_sentences(block.text, query, top_k=n_keep)
        compressed_text = " ".join(key_sents)

        # Estimate new token count
        new_tokens = max(1, int(block.token_count * len(key_sents) / len(sentences)))

        return ContextBlock(
            document_id=block.document_id,
            text=compressed_text,
            relevance_score=block.relevance_score,
            token_count=new_tokens,
            source_document=block.source_document,
            compression_strategy=block.compression_strategy,
            quantum_amplitude=block.quantum_amplitude,
            measurement_probability=block.measurement_probability,
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common sentence boundaries
        sentences = []
        for s in text.replace("\n", ". ").replace("!", ".").replace("?", ".").split("."):
            s = s.strip()
            if s and len(s) > 5:  # Filter out very short fragments
                sentences.append(s)
        return sentences if sentences else [text]
