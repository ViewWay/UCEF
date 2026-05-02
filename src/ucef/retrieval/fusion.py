"""
Multi-Strategy Fusion — Reciprocal Rank Fusion & Weighted Score Aggregation

Combines results from multiple retrieval strategies (hyperbolic, quantum,
keyword) into a unified ranked list. Implements standard fusion methods
from information retrieval literature.

References:
    - Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and
      Individual Rank Learning Methods", SIGIR 2009
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ucef.core.types import Document


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining ranked lists.

    RRF score for document d:
        RRF(d) = Σ_{r ∈ rankings} 1 / (k + r(d))

    where r(d) is the rank of d in ranking r, and k is a constant
    (default 60) that mitigates the impact of high rankings from
    a single outlier list.

    Reference: Cormack et al. (2009)
    """

    def __init__(self, k: int = 60) -> None:
        self._k = k

    def fuse(
        self,
        rankings: Sequence[List[Tuple[Document, float]]],
        weights: Optional[Sequence[float]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Fuse multiple ranked lists into a single ranking.

        Args:
            rankings: Multiple ranked lists, each as (doc, score) tuples.
            weights: Optional per-list weights. If None, equal weighting.

        Returns:
            Fused ranking as (document, rrf_score) tuples.
        """
        if not rankings:
            return []

        if weights is None:
            weights = [1.0] * len(rankings)
        elif len(weights) != len(rankings):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of rankings ({len(rankings)})"
            )

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for ranking, weight in zip(rankings, weights):
            for rank, (doc, _original_score) in enumerate(ranking, start=1):
                doc_map[doc.id] = doc
                contribution = weight / (self._k + rank)
                rrf_scores[doc.id] = rrf_scores.get(doc.id, 0.0) + contribution

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

        return [(doc_map[doc_id], rrf_scores[doc_id]) for doc_id in sorted_ids]


class WeightedScoreFusion:
    """
    Weighted score aggregation for combining retrieval results.

    Final score = Σᵢ wᵢ · scoreᵢ(doc)

    Supports score normalization per list before fusion.
    """

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            weights: Per-strategy weights. None means equal weighting.
            normalize: Whether to min-max normalize scores per list.
        """
        self._weights = weights
        self._normalize = normalize

    def fuse(
        self,
        result_lists: Sequence[List[Tuple[Document, float]]],
    ) -> List[Tuple[Document, float]]:
        """
        Fuse multiple scored lists via weighted aggregation.

        Args:
            result_lists: Multiple lists of (document, score) tuples.
                         Scores should be comparable across lists.

        Returns:
            Fused ranking as (document, weighted_score) tuples.
        """
        if not result_lists:
            return []

        weights = self._weights or [1.0] * len(result_lists)

        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for result_list, weight in zip(result_lists, weights):
            if not result_list:
                continue

            # Extract scores for normalization
            scores = np.array([s for _, s in result_list], dtype=np.float64)

            if self._normalize and len(scores) > 1:
                # Min-max normalization to [0, 1]
                s_min, s_max = scores.min(), scores.max()
                if s_max - s_min > 1e-10:
                    scores = (scores - s_min) / (s_max - s_min)
                else:
                    scores = np.ones_like(scores) * 0.5

            for i, (doc, _) in enumerate(result_list):
                doc_map[doc.id] = doc
                contribution = weight * float(scores[i])
                doc_scores[doc.id] = doc_scores.get(doc.id, 0.0) + contribution

        sorted_ids = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)

        return [(doc_map[doc_id], doc_scores[doc_id]) for doc_id in sorted_ids]


class HybridFusion:
    """
    Hybrid fusion combining RRF (rank-based) and weighted scores.

    Useful when different retrieval strategies produce incomparable scores.
    Uses RRF for rank-based signals and weighted fusion for score signals.

    final_score = α · rrf_normalized + (1 - α) · weighted_normalized
    """

    def __init__(self, alpha: float = 0.5, rrf_k: int = 60) -> None:
        """
        Args:
            alpha: Weight for RRF component (0.0 = pure weighted, 1.0 = pure RRF).
            rrf_k: RRF constant k.
        """
        self._alpha = alpha
        self._rrf = ReciprocalRankFusion(k=rrf_k)
        self._weighted = WeightedScoreFusion(normalize=True)

    def fuse(
        self,
        result_lists: Sequence[List[Tuple[Document, float]]],
    ) -> List[Tuple[Document, float]]:
        """Combine RRF and weighted fusion."""
        rrf_results = self._rrf.fuse(result_lists)
        weighted_results = self._weighted.fuse(result_lists)

        # Normalize both score distributions to [0, 1]
        rrf_scores = {doc.id: score for doc, score in rrf_results}
        weighted_scores = {doc.id: score for doc, score in weighted_results}
        doc_map: Dict[str, Document] = {}

        all_ids = set(rrf_scores.keys()) | set(weighted_scores.keys())
        for doc, _ in rrf_results + weighted_results:
            doc_map[doc.id] = doc

        def normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
            vals = list(score_dict.values())
            if not vals:
                return score_dict
            vmin, vmax = min(vals), max(vals)
            if vmax - vmin < 1e-10:
                return {k: 0.5 for k in score_dict}
            return {k: (v - vmin) / (vmax - vmin) for k, v in score_dict.items()}

        rrf_norm = normalize_scores(rrf_scores)
        weighted_norm = normalize_scores(weighted_scores)

        final_scores: Dict[str, float] = {}
        for doc_id in all_ids:
            r = rrf_norm.get(doc_id, 0.0)
            w = weighted_norm.get(doc_id, 0.0)
            final_scores[doc_id] = self._alpha * r + (1.0 - self._alpha) * w

        sorted_ids = sorted(final_scores.keys(), key=lambda d: final_scores[d], reverse=True)

        return [(doc_map[doc_id], final_scores[doc_id]) for doc_id in sorted_ids]
