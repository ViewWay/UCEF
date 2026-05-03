"""
Quantum Field Model — Renormalization Group for Multi-Scale Context Compression

Inspired by Wilson's renormalization group (RG), implements multi-scale
coarse-graining of context:

    Scale 0 (finest):    Full text preserved
    Scale 1 (sentences): Key sentence extraction
    Scale 2 (paragraph): Paragraph-level summaries

Analogy to physics:
    - UV cutoff (Λ_max) = token budget
    - Effective action S_eff[φ] = compressed context
    - Scale transformation R_s = summarization operation
    - Coupling flow g(s) = relevance at different scales

References:
    - Wilson, "The Renormalization Group: Critical Phenomena and the Kondo
      Problem", Rev. Mod. Phys. 1975
    - Goldenfeld, "Lectures on Phase Transitions and the Renormalization
      Group", Westview Press 1992
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from ucef.core.types import CompressionResult, ContextBlock


class RenormalizationGroup:
    """
    Multi-scale context compression via renormalization group flow.

    Documents are compressed at progressively coarser scales:
        - Scale 0: No compression (full detail)
        - Scale 1: Sentence-level extraction
        - Scale 2: Block-level summarization (top sentences only)

    The number of compression scales applied to each block depends on
    its relevance: high-relevance blocks keep more detail (fewer scales),
    while low-relevance blocks are compressed more aggressively.
    """

    def __init__(self, n_scales: int = 3) -> None:
        """
        Args:
            n_scales: Number of coarse-graining scales (1-5).
        """
        self._n_scales = max(1, min(n_scales, 5))

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def multiscale_compress(
        self,
        blocks: List[ContextBlock],
        budget: int,
        query: str = "",
    ) -> Tuple[List[ContextBlock], CompressionResult]:
        """
        Apply multi-scale compression to blocks.

        Each block is assigned a compression scale based on its relevance:
            - Top relevance → Scale 0 (no compression)
            - Medium → Scale 1 (sentence extraction)
            - Low → Scale 2 (aggressive extraction)

        Then blocks are selected within budget.
        """
        if not blocks:
            return [], CompressionResult(original_tokens=0, compressed_tokens=0)

        original_tokens = sum(b.token_count for b in blocks)

        # Compute relevance flow for each block
        relevance_scores = self.relevance_flow(blocks, query)

        # Assign compression scales based on relevance percentile
        sorted_indices = sorted(range(len(blocks)),
                                key=lambda i: relevance_scores[i],
                                reverse=True)

        n = len(blocks)
        compressed_blocks: List[ContextBlock] = [None] * n  # type: ignore

        for rank, idx in enumerate(sorted_indices):
            block = blocks[idx]
            rel = relevance_scores[idx]

            # Determine compression scale based on rank
            if rank < n // 3:
                # Top third: light or no compression
                scale = 0 if rel > 0.5 else 1
            elif rank < 2 * n // 3:
                # Middle third: sentence extraction
                scale = 1
            else:
                # Bottom third: aggressive extraction
                scale = min(2, self._n_scales - 1)

            compressed_blocks[idx] = self._apply_scale(block, scale, query)

        # Filter out None and empty blocks
        valid_blocks = [b for b in compressed_blocks if b and b.text.strip()]

        # Select within budget by relevance
        valid_blocks.sort(key=lambda b: b.relevance_score, reverse=True)
        selected: List[ContextBlock] = []
        total_tokens = 0
        for block in valid_blocks:
            if total_tokens + block.token_count <= budget:
                selected.append(block)
                total_tokens += block.token_count

        result = CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=total_tokens,
        )

        return selected, result

    def relevance_flow(
        self,
        blocks: List[ContextBlock],
        query: str,
    ) -> List[float]:
        """
        Compute relevance "flow" — how relevance changes under coarse-graining.

        Analogous to coupling constant flow in RG:
            g(s) = g₀ · s^(-Δ)
        where Δ is the scaling dimension.

        Blocks with high flow maintain relevance at all scales.
        Blocks with low flow lose relevance quickly under compression.
        """
        query_words = set(query.lower().split()) if query else set()

        scores: List[float] = []
        for block in blocks:
            # Base relevance
            base = block.relevance_score

            # Query overlap
            block_words = set(block.text.lower().split())
            overlap = len(query_words & block_words) / max(len(query_words), 1) if query_words else 0.0

            # Information density
            density = block.information_density

            # Flow score: weighted combination
            # High flow = stays relevant even when compressed
            flow = 0.4 * base + 0.35 * overlap + 0.25 * density
            scores.append(flow)

        return scores

    def coarse_grain(
        self,
        text: str,
        target_ratio: float,
        query: str = "",
    ) -> str:
        """
        Single coarse-graining operation: extract core content from text.

        Args:
            text: Input text.
            target_ratio: Fraction of content to retain (0-1).
            query: Optional query for relevance weighting.

        Returns:
            Coarse-grained text.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text

        n_keep = max(1, int(len(sentences) * target_ratio))

        # Score sentences
        query_words = set(query.lower().split()) if query else set()
        scored: List[Tuple[int, str, float]] = []

        for i, sent in enumerate(sentences):
            sent_words = set(sent.lower().split())

            # Query relevance
            overlap = len(query_words & sent_words) / max(len(query_words), 1) if query_words else 0.0

            # Position (first/last sentences carry more weight)
            pos = 1.0
            if i == 0:
                pos = 1.3
            elif i == len(sentences) - 1:
                pos = 1.2

            # Self-information (prefer diverse sentences)
            freq = Counter(sent_words)
            n = len(sent_words) or 1
            info = -sum((c / n) * math.log2(max(c / n, 1e-10)) for c in freq.values())

            score = 0.5 * overlap * pos + 0.3 * info + 0.2 * (len(sent_words) / 20.0)
            scored.append((i, sent, score))

        # Select and maintain order
        scored.sort(key=lambda x: x[2], reverse=True)
        top = sorted(scored[:n_keep], key=lambda x: x[0])

        return " ".join(s for _, s, _ in top)

    # ──────────────────────────────────────────────────────────────────────
    # Internal Methods
    # ──────────────────────────────────────────────────────────────────────

    def _apply_scale(
        self,
        block: ContextBlock,
        scale: int,
        query: str,
    ) -> ContextBlock:
        """Apply compression at the specified scale level."""
        if scale == 0:
            # Scale 0: No compression
            return block

        # Determine retention ratio based on scale
        ratios = {0: 1.0, 1: 0.6, 2: 0.35, 3: 0.2, 4: 0.1}
        ratio = ratios.get(scale, 0.3)

        compressed_text = self.coarse_grain(block.text, ratio, query)

        # Estimate new token count
        original_words = len(block.text.split())
        compressed_words = len(compressed_text.split())
        if original_words > 0:
            new_tokens = max(1, int(block.token_count * compressed_words / original_words))
        else:
            new_tokens = block.token_count

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
        sentences = []
        for s in text.replace("\n", ". ").replace("!", ".").replace("?", ".").split("."):
            s = s.strip()
            if s and len(s) > 3:
                sentences.append(s)
        return sentences if sentences else [text]
