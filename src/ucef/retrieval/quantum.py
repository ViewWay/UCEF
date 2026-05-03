"""
Quantum Selector — Density Matrix Construction & Measurement-Based Context Selection

Implements quantum-inspired context selection using the mathematical framework
of quantum probability theory. Documents are represented as quantum states,
and a query acts as a measurement operator that collapses the superposition
to the most relevant contexts.

Key innovations:
- Superposition: parallel consideration of all candidate contexts
- Entanglement: captures inter-document correlations via off-diagonal density matrix
- Interference: constructive/destructive filtering of relevant/irrelevant contexts

References:
    - van Rijsbergen, "The Geometry of Information Retrieval", CUP, 2004
    - Zuccon et al., "The Quantum Probability Ranking Principle", ICTIR 2009
    - QISA, "Quantum-Inspired Self-Attention", arXiv:2603.03318, 2024
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ucef.core.config import QuantumConfig
from ucef.core.types import (
    ContextBlock,
    DensityMatrix,
    Document,
    QuantumState,
    TokenBudget,
)


class QuantumSelector:
    """
    Selects context using quantum-inspired measurement.

    Pipeline:
    1. Construct quantum state from relevance scores (superposition)
    2. Compute density matrix (captures pairwise correlations)
    3. Apply query as measurement operator
    4. Collapse to selected contexts (measurement)

    Usage:
        selector = QuantumSelector(config)
        blocks = selector.select(scored_docs, query_vec, budget)
    """

    def __init__(self, config: Optional[QuantumConfig] = None) -> None:
        self._config = config or QuantumConfig()

    def select(
        self,
        scored_documents: List[Tuple[Document, float]],
        query_weights: Optional[NDArray[np.float64]] = None,
        budget: Optional[TokenBudget] = None,
    ) -> List[ContextBlock]:
        """
        Select context blocks via quantum measurement.

        The selection process:
        1. Build superposition |ψ⟩ = Σᵢ √pᵢ |docᵢ⟩ from relevance scores
        2. Construct density matrix ρ = |ψ⟩⟨ψ| (+ entanglement corrections)
        3. Apply query as measurement: compute quantum similarity S(q, ρ)
        4. Collapse: select top-k by measurement probability, respecting budget

        Args:
            scored_documents: List of (document, relevance_score) tuples.
            query_weights: Optional query-specific weighting vector.
            budget: Token budget constraint.

        Returns:
            Selected ContextBlock list.
        """
        if not scored_documents:
            return []

        n = len(scored_documents)

        # Step 1: Construct quantum state from relevance scores
        state = self._build_superposition(scored_documents)

        # Step 2: Construct density matrix (with optional entanglement)
        rho = self._build_density_matrix(state, scored_documents)

        # Step 3: Apply query measurement
        if query_weights is not None:
            measurement_probs = self._measure_with_query(rho, query_weights)
        else:
            measurement_probs = state.probabilities

        # Step 3.5: Apply interference filtering if enabled
        if self._config.use_interference:
            measurement_probs = self._apply_interference(
                measurement_probs, state, scored_documents
            )

        # Step 4: Collapse — select top-k respecting budget
        blocks = self._collapse_to_blocks(
            scored_documents, measurement_probs, budget or self._default_budget()
        )

        return blocks

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Superposition Construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_superposition(
        self,
        scored_documents: List[Tuple[Document, float]],
    ) -> QuantumState:
        """
        Build quantum superposition from relevance scores.

        |ψ⟩ = Σᵢ √pᵢ |docᵢ⟩

        where pᵢ = score_i / Σ score_j (Born rule: P(i) = |αᵢ|² = pᵢ)

        Initial amplitude strategy is determined by config:
        - "equal": uniform superposition (no prior knowledge)
        - "relevance_weighted": weight by relevance scores
        - "entropy_weighted": weight by information entropy
        """
        n = len(scored_documents)
        labels = [doc.id for doc, _ in scored_documents]

        if self._config.initial_amplitude == "equal":
            return QuantumState.equal_superposition(n, labels)

        scores = np.array([s for _, s in scored_documents], dtype=np.float64)

        if self._config.initial_amplitude == "entropy_weighted":
            scores = self._compute_entropy_weights(scored_documents)

        # Default: relevance_weighted
        scores = np.clip(scores, 1e-6, None)
        probs = scores / scores.sum()

        return QuantumState.from_probabilities(probs, labels)

    def _compute_entropy_weights(
        self,
        scored_documents: List[Tuple[Document, float]],
    ) -> NDArray[np.float64]:
        """
        Compute weights based on information entropy of each document.

        H(doc) = -Σ p(word) log p(word)

        Higher entropy → more diverse information → higher weight.
        """
        weights = np.zeros(len(scored_documents), dtype=np.float64)

        for i, (doc, score) in enumerate(scored_documents):
            if doc.token_count > 0:
                # Approximate entropy: more unique words → higher entropy
                words = doc.text.lower().split()
                if words:
                    from collections import Counter
                    counts = Counter(words)
                    total = len(words)
                    entropy = -sum(
                        (c / total) * np.log(c / total + 1e-10)
                        for c in counts.values()
                    )
                    weights[i] = entropy * score  # Combine with relevance
                else:
                    weights[i] = score * 0.5
            else:
                weights[i] = score * 0.5

        return weights

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Density Matrix Construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_density_matrix(
        self,
        state: QuantumState,
        scored_documents: List[Tuple[Document, float]],
    ) -> DensityMatrix:
        """
        Build density matrix from quantum state.

        For pure states: ρ = |ψ⟩⟨ψ|

        For mixed states with entanglement: adds off-diagonal corrections
        based on document-document similarity (entanglement).

        Entanglement between documents i,j is modeled as:
            ρᵢⱼ += entanglement_strength * similarity(i,j)
        """
        rho = DensityMatrix.from_pure_state(state)

        if not self._config.use_interference:
            return rho

        # Add entanglement corrections based on text overlap
        n = len(scored_documents)
        matrix = rho.matrix.copy()

        for i in range(n):
            for j in range(i + 1, n):
                doc_i, _ = scored_documents[i]
                doc_j, _ = scored_documents[j]

                # Compute text-based similarity (entanglement proxy)
                similarity = self._text_similarity(doc_i, doc_j)

                if similarity > self._config.entanglement_threshold:
                    # Entangled: add coherent off-diagonal contribution
                    entanglement = similarity * 0.1  # Scale factor
                    matrix[i, j] += entanglement
                    matrix[j, i] += np.conj(entanglement)

        return DensityMatrix(matrix=matrix)

    @staticmethod
    def _text_similarity(doc_a: Document, doc_b: Document) -> float:
        """
        Compute Jaccard similarity between two documents.

        J(A, B) = |A ∩ B| / |A ∪ B|
        """
        words_a = set(doc_a.text.lower().split())
        words_b = set(doc_b.text.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        if union == 0:
            return 0.0

        return intersection / union

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Query Measurement
    # ──────────────────────────────────────────────────────────────────────

    def _measure_with_query(
        self,
        rho: DensityMatrix,
        query_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Apply query as measurement operator on density matrix.

        S(q, c) = ⟨q|ρ|c⟩ for each basis state c.

        The query vector encodes the user's information need.
        The density matrix captures document correlations.
        Together they produce a measurement probability distribution.
        """
        if len(query_weights) != rho.dimension:
            # Dimension mismatch — fall back to diagonal (classical)
            diag = np.real(np.diag(rho.matrix))
            return diag / max(diag.sum(), 1e-10)

        query_vec = query_weights.astype(np.complex128)
        similarities = rho.quantum_similarity(query_vec)

        # Convert to probability distribution
        probs = similarities.copy()
        probs = np.clip(probs, 1e-10, None)  # Ensure non-negative
        total = probs.sum()
        if total > 0:
            probs = probs / total

        return probs

    # ──────────────────────────────────────────────────────────────────────
    # Step 3.5: Interference Filtering
    # ──────────────────────────────────────────────────────────────────────

    def _apply_interference(
        self,
        probabilities: NDArray[np.float64],
        state: QuantumState,
        scored_documents: List[Tuple[Document, float]],
    ) -> NDArray[np.float64]:
        """
        Apply quantum interference to filter context.

        Constructive interference: boost documents that align with query theme.
        Destructive interference: suppress contradictory or redundant documents.

        Interference pattern:
            I(i) = |Σⱼ αᵢ αⱼ* cos(θᵢⱼ)|²

        where θᵢⱼ is the phase difference (similarity angle) between docs i, j.
        """
        n = len(scored_documents)
        amplitudes = state.amplitudes

        # Precompute pairwise similarity matrix (vectorized Jaccard)
        word_sets = [set(doc.text.lower().split()) for doc, _ in scored_documents]
        sim_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                if not word_sets[i] or not word_sets[j]:
                    continue
                inter = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                sim = inter / union if union > 0 else 0.0
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        # Phase matrix: similar docs have constructive interference (phase ≈ 0)
        phase_matrix = (1.0 - sim_matrix) * np.pi / 2.0
        np.fill_diagonal(phase_matrix, 0.0)

        # Compute interference: I(i) = Σ_j Re(αᵢ αⱼ* cos(θᵢⱼ))
        amp_outer = np.outer(amplitudes, np.conj(amplitudes))
        interference_terms = np.real(amp_outer * np.cos(phase_matrix))
        interference = 1.0 + np.sum(interference_terms, axis=1) - np.diag(interference_terms)

        # Interference modulation: probabilities * |interference|
        modulated = probabilities * np.abs(interference)
        modulated = np.clip(modulated, 1e-10, None)

        # Renormalize
        total = modulated.sum()
        if total > 0:
            modulated = modulated / total

        return modulated

    # ──────────────────────────────────────────────────────────────────────
    # Step 4: Measurement Collapse
    # ──────────────────────────────────────────────────────────────────────

    def _collapse_to_blocks(
        self,
        scored_documents: List[Tuple[Document, float]],
        probabilities: NDArray[np.float64],
        budget: TokenBudget,
    ) -> List[ContextBlock]:
        """
        Collapse the quantum state to selected context blocks.

        Measurement methods:
        - "argmax": select highest probability items
        - "sampling": probabilistic sampling
        - "top_k": select top-k by probability, respecting budget
        """
        method = self._config.measurement_method

        if method == "sampling":
            indices = self._sample_collapse(probabilities)
        elif method == "top_k":
            indices = self._topk_collapse(probabilities)
        else:  # "argmax"
            indices = self._argmax_collapse(probabilities)

        # Apply budget constraint
        blocks: List[ContextBlock] = []
        total_tokens = 0
        max_tokens = budget.available_for_retrieval

        for idx in indices:
            if idx >= len(scored_documents):
                continue
            doc, score = scored_documents[idx]
            doc_tokens = doc.estimate_tokens()

            if total_tokens + doc_tokens > max_tokens:
                continue  # Over budget — skip

            block = ContextBlock(
                document_id=doc.id,
                text=doc.text,
                relevance_score=float(score),
                token_count=doc_tokens,
                quantum_amplitude=complex(probabilities[idx] ** 0.5, 0),
                measurement_probability=float(probabilities[idx]),
            )
            blocks.append(block)
            total_tokens += doc_tokens

        return blocks

    def _argmax_collapse(self, probabilities: NDArray[np.float64]) -> NDArray[np.intp]:
        """Select items by descending probability."""
        return np.argsort(probabilities)[::-1]

    def _topk_collapse(self, probabilities: NDArray[np.float64]) -> NDArray[np.intp]:
        """Select top-k items by probability."""
        k = min(self._config.top_k_measurements, len(probabilities))
        return np.argsort(probabilities)[::-1][:k]

    def _sample_collapse(self, probabilities: NDArray[np.float64]) -> NDArray[np.intp]:
        """
        Probabilistic sampling — simulates quantum measurement randomness.

        Each "measurement" draws a sample according to the Born rule
        probability distribution.
        """
        n_samples = min(self._config.top_k_measurements, len(probabilities))
        # Sample without replacement using probabilities
        indices = np.random.choice(
            len(probabilities),
            size=min(n_samples, len(probabilities)),
            replace=False,
            p=probabilities,
        )
        return indices

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _default_budget() -> TokenBudget:
        """Create a default token budget for when none is provided."""
        return TokenBudget.from_context_window(
            context_window=128_000,
            system_tokens=500,
            conversation_tokens=1000,
            response_tokens=2000,
        )
