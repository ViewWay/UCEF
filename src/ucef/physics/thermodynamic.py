"""
Thermodynamic Model — Free Energy Minimization for Context Selection

Maps context selection to a free energy minimization problem from
statistical mechanics:

    F = E - T · S

where:
    E(context) = "energy" (irrelevance to query)
    S(context) = information entropy (diversity)
    T = temperature (controls exploration vs exploitation)

Temperature scheduling:
    - High T: maximize diversity (exploration)
    - Low T: minimize energy (exploitation)
    - Annealing: T decreases over compression rounds

References:
    - Jaynes, "Information Theory and Statistical Mechanics", Phys. Rev. 1957
    - Kirkpatrick et al., "Optimization by Simulated Annealing", Science 1983
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from ucef.core.types import ContextBlock


class ThermodynamicModel:
    """
    Thermodynamic context selection model.

    Selects context blocks by minimizing free energy F = E - T·S,
    balancing relevance (low energy) with diversity (high entropy).

    Usage:
        model = ThermodynamicModel(temperature=1.0, cooling_rate=0.95)
        selected = model.select_by_free_energy(blocks, query, budget)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
    ) -> None:
        self._temperature = temperature
        self._cooling_rate = cooling_rate
        self._min_temperature = min_temperature
        self._initial_temperature = temperature

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def temperature(self) -> float:
        """Current temperature parameter."""
        return self._temperature

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def select_by_free_energy(
        self,
        blocks: List[ContextBlock],
        query: str,
        budget: int,
        n_rounds: int = 5,
    ) -> List[ContextBlock]:
        """
        Select blocks by simulated annealing on free energy.

        Each round:
            1. Compute F(block) = E(block) - T · S(block)
            2. Select top blocks by lowest F
            3. Anneal: T *= cooling_rate

        Args:
            blocks: Candidate context blocks.
            query: User query for energy computation.
            budget: Maximum total tokens.
            n_rounds: Number of annealing rounds.

        Returns:
            Selected context blocks.
        """
        if not blocks:
            return []

        # Reset temperature
        self._temperature = self._initial_temperature

        query_words = set(query.lower().split()) if query else set()

        best_selection: List[ContextBlock] = []
        best_score = float("inf")

        for _ in range(n_rounds):
            # Compute free energy for each block
            selected: List[ContextBlock] = []
            selected_words: Set[str] = set()
            total_tokens = 0

            # Score all blocks
            scored: List[Tuple[ContextBlock, float]] = []
            for block in blocks:
                energy = self.energy(block, query_words)
                entropy = self.entropy_contribution(block, selected_words)
                fe = self.free_energy(energy, entropy)
                scored.append((block, fe))

            # Sort by free energy (lower is better)
            scored.sort(key=lambda x: x[1])

            # Greedy selection within budget
            for block, _ in scored:
                if total_tokens + block.token_count <= budget:
                    selected.append(block)
                    total_tokens += block.token_count
                    selected_words.update(block.text.lower().split())

            # Evaluate total free energy of selection
            total_fe = sum(
                self.free_energy(
                    self.energy(b, query_words),
                    self.entropy_contribution(b, selected_words),
                )
                for b in selected
            )

            if total_fe < best_score:
                best_score = total_fe
                best_selection = selected[:]

            # Anneal
            self.anneal()

        return best_selection

    def energy(self, block: ContextBlock, query_words: Set[str]) -> float:
        """
        Compute energy (irrelevance) of a block relative to query.

        E = 1 - relevance_score
        Low energy = highly relevant.
        """
        if not query_words:
            return 1.0 - block.relevance_score

        block_words = set(block.text.lower().split())
        overlap = len(query_words & block_words) / len(query_words)
        # Energy is inverse of relevance
        return 1.0 - overlap

    def entropy_contribution(
        self,
        block: ContextBlock,
        already_selected_words: Set[str],
    ) -> float:
        """
        Compute marginal entropy contribution of adding this block.

        S = |new_words| / |total_words|
        Higher entropy = more novel information.
        """
        block_words = set(block.text.lower().split())
        if not block_words:
            return 0.0

        # Novel words not in already-selected set
        novel = block_words - already_selected_words
        if not already_selected_words:
            return 1.0  # First block is maximally novel

        return len(novel) / max(len(block_words), 1)

    def free_energy(self, energy: float, entropy: float) -> float:
        """
        Compute free energy: F = E - T · S

        Lower F = better selection (relevant + diverse).
        """
        return energy - self._temperature * entropy

    def anneal(self) -> None:
        """
        Temperature annealing: T = max(T * cooling_rate, min_temperature).
        """
        self._temperature = max(
            self._temperature * self._cooling_rate,
            self._min_temperature,
        )

    def reset(self) -> None:
        """Reset temperature to initial value."""
        self._temperature = self._initial_temperature

    def boltzmann_probabilities(
        self,
        blocks: List[ContextBlock],
        query_words: Set[str],
    ) -> List[float]:
        """
        Compute Boltzmann selection probabilities.

        P(i) = exp(-β · E_i) / Z
        where β = 1/T, Z = Σ exp(-β · E_j)

        Useful for stochastic (sampling-based) selection.
        """
        beta = 1.0 / max(self._temperature, 1e-10)
        energies = [self.energy(b, query_words) for b in blocks]

        # Boltzmann weights
        log_weights = [-beta * e for e in energies]
        max_log = max(log_weights)  # For numerical stability
        weights = [math.exp(lw - max_log) for lw in log_weights]

        z = sum(weights)
        if z < 1e-10:
            return [1.0 / len(blocks)] * len(blocks)

        return [w / z for w in weights]
