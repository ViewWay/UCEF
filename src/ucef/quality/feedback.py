"""
Quality Feedback Loop — Automatic Quality Refinement Cycle

Implements a closed-loop feedback system that detects low-quality results
and automatically triggers refinement actions (re-retrieve, re-select,
re-compress) until quality meets the threshold or max iterations reached.

Refinement strategies:
    1. Expand retrieval: increase top_k to find more candidates
    2. Relax quantum selection: allow more blocks through
    3. Reduce compression: use lighter strategy to preserve more content
    4. Full re-query: start over with adjusted parameters

References:
    - UCEF Architecture: docs/api/architecture.md
    - Quality weights: ucef.core.config.QualityConfig
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence

from ucef.core.types import (
    CompressionStrategy,
    ContextBlock,
    QueryResult,
    TokenBudget,
)


class RefinementAction(Enum):
    """Actions the feedback loop can take to improve quality."""
    EXPAND_RETRIEVAL = auto()    # Increase top_k
    RELAX_SELECTION = auto()     # Allow more blocks through quantum filter
    LIGHTEN_COMPRESSION = auto() # Switch to less aggressive compression
    FULL_REQUERY = auto()        # Start over with adjusted parameters


@dataclass
class RefinementStep:
    """Record of a single refinement iteration."""
    iteration: int
    action: RefinementAction
    quality_before: float
    quality_after: float
    improvement: float
    elapsed_ms: float


@dataclass
class FeedbackResult:
    """Result of the feedback refinement cycle."""
    final_result: QueryResult
    iterations: int
    steps: List[RefinementStep] = field(default_factory=list)
    converged: bool = False
    total_refinement_ms: float = 0.0

    @property
    def total_improvement(self) -> float:
        """Total quality improvement across all iterations."""
        if not self.steps:
            return 0.0
        return self.steps[-1].quality_after - self.steps[0].quality_before


class QualityFeedbackLoop:
    """
    Closed-loop quality refinement for UCEF queries.

    When initial query quality is below threshold, iteratively applies
    refinement strategies until quality is acceptable or max iterations
    are exhausted.

    Usage:
        loop = QualityFeedbackLoop(max_iterations=3, target_quality=0.75)
        result = await loop.refine(
            initial_result=result,
            requery_fn=system.query,
        )
    """

    def __init__(
        self,
        max_iterations: int = 3,
        target_quality: float = 0.75,
        min_improvement: float = 0.02,
        initial_top_k: int = 10,
        top_k_multiplier: float = 1.5,
    ) -> None:
        self._max_iterations = max_iterations
        self._target_quality = target_quality
        self._min_improvement = min_improvement
        self._initial_top_k = initial_top_k
        self._top_k_multiplier = top_k_multiplier
        self._requery_timeout_s: float = 30.0  # Timeout for each requery call

    async def refine(
        self,
        initial_result: QueryResult,
        requery_fn: Callable[..., Any],
        *,
        query: str = "",
        quality_threshold: Optional[float] = None,
    ) -> FeedbackResult:
        """
        Run feedback refinement cycle.

        Args:
            initial_result: The initial query result to refine.
            requery_fn: Async callable that re-executes the query pipeline.
                Signature: async (query, top_k, quality_threshold, **kwargs) -> QueryResult
            query: Original query string.
            quality_threshold: Override target quality for this call.

        Returns:
            FeedbackResult with final (possibly refined) result and iteration history.
        """
        threshold = quality_threshold or self._target_quality
        start_time = time.monotonic()

        current_result = initial_result
        steps: List[RefinementStep] = []

        # If quality is already good enough, return immediately
        if current_result.overall_quality >= threshold:
            return FeedbackResult(
                final_result=current_result,
                iterations=0,
                steps=[],
                converged=True,
                total_refinement_ms=(time.monotonic() - start_time) * 1000,
            )

        current_top_k = self._initial_top_k
        strategy_chain = [
            RefinementAction.EXPAND_RETRIEVAL,
            RefinementAction.LIGHTEN_COMPRESSION,
            RefinementAction.FULL_REQUERY,
        ]

        for i in range(self._max_iterations):
            action = strategy_chain[i % len(strategy_chain)]
            step_start = time.monotonic()

            # Determine parameters based on action
            kwargs: Dict[str, Any] = {}
            if action == RefinementAction.EXPAND_RETRIEVAL:
                current_top_k = int(current_top_k * self._top_k_multiplier)
                kwargs["top_k"] = current_top_k
            elif action == RefinementAction.LIGHTEN_COMPRESSION:
                kwargs["top_k"] = current_top_k
            elif action == RefinementAction.FULL_REQUERY:
                current_top_k = int(current_top_k * self._top_k_multiplier * 1.5)
                kwargs["top_k"] = current_top_k
            else:
                kwargs["top_k"] = current_top_k

            # Re-execute query with timeout
            try:
                new_result = await asyncio.wait_for(
                    requery_fn(
                        query, top_k=kwargs.get("top_k"),
                        quality_threshold=threshold,
                    ),
                    timeout=self._requery_timeout_s,
                )
            except asyncio.TimeoutError:
                # Timeout — stop refining
                break
            except Exception:
                # If requery fails, keep current result
                break

            elapsed = (time.monotonic() - step_start) * 1000

            step = RefinementStep(
                iteration=i + 1,
                action=action,
                quality_before=current_result.overall_quality,
                quality_after=new_result.overall_quality,
                improvement=new_result.overall_quality - current_result.overall_quality,
                elapsed_ms=elapsed,
            )
            steps.append(step)

            # Update current result
            current_result = new_result

            # Check convergence
            if current_result.overall_quality >= threshold:
                return FeedbackResult(
                    final_result=current_result,
                    iterations=i + 1,
                    steps=steps,
                    converged=True,
                    total_refinement_ms=(time.monotonic() - start_time) * 1000,
                )

            # Check for stagnation — if improvement is too small, stop
            if step.improvement < self._min_improvement and i > 0:
                break

        return FeedbackResult(
            final_result=current_result,
            iterations=len(steps),
            steps=steps,
            converged=current_result.overall_quality >= threshold,
            total_refinement_ms=(time.monotonic() - start_time) * 1000,
        )

    def diagnose(self, result: QueryResult) -> List[RefinementAction]:
        """
        Diagnose quality issues and recommend refinement actions.

        Analyzes which quality dimension is weakest and returns
        ordered actions most likely to improve that dimension.

        Returns:
            List of recommended RefinementActions in priority order.
        """
        actions: List[RefinementAction] = []

        # Low relevance → expand retrieval to find better candidates
        if result.relevance_score < 0.5:
            actions.append(RefinementAction.EXPAND_RETRIEVAL)

        # Low completeness → we're missing relevant info
        if result.completeness_score < 0.5:
            actions.append(RefinementAction.EXPAND_RETRIEVAL)
            actions.append(RefinementAction.LIGHTEN_COMPRESSION)

        # Low coherence → too much compression breaking flow
        if result.coherence_score < 0.4:
            actions.append(RefinementAction.LIGHTEN_COMPRESSION)

        # Everything is low → full requery
        if result.overall_quality < 0.3:
            actions.append(RefinementAction.FULL_REQUERY)

        # Deduplicate while preserving order
        seen = set()
        unique_actions = []
        for action in actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)

        return unique_actions or [RefinementAction.EXPAND_RETRIEVAL]
