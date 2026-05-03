"""
Quality Monitor — Real-time Quality Tracking and Alerting

Monitors query results in real-time, tracks quality trends, and emits
alerts when quality drops below thresholds.

Metrics tracked:
    - Per-query: relevance, completeness, coherence, accuracy, overall
    - Rolling window: mean, min, p95 of each metric
    - Cumulative: total queries, average quality, quality distribution

References:
    - UCEF Architecture: docs/api/architecture.md
    - Quality weights: ucef.core.config.QualityConfig
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from ucef.core.types import QualityIssue, QualityIssueType, QueryResult


@dataclass
class QualitySample:
    """A single quality measurement sample."""
    timestamp: float
    query: str
    relevance: float
    completeness: float
    coherence: float
    accuracy: float
    overall: float
    n_blocks: int
    total_tokens: int
    latency_ms: float


class QualityMonitor:
    """
    Real-time quality monitor for UCEF query results.

    Tracks quality metrics over a rolling window and provides:
    - Quality trend analysis
    - Anomaly detection (sudden quality drops)
    - Quality distribution statistics

    Usage:
        monitor = QualityMonitor(window_size=100)
        monitor.record(result)
        if monitor.is_quality_degraded():
            alert = monitor.get_alert()
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_threshold: float = 0.6,
        degradation_threshold: float = 0.15,
    ) -> None:
        self._window_size = window_size
        self._alert_threshold = alert_threshold
        self._degradation_threshold = degradation_threshold

        self._samples: Deque[QualitySample] = deque(maxlen=window_size)
        self._total_queries = 0
        self._total_quality_sum = 0.0
        self._issues: List[QualityIssue] = []

    # ──────────────────────────────────────────────────────────────────────
    # Recording
    # ──────────────────────────────────────────────────────────────────────

    def record(self, result: QueryResult) -> None:
        """
        Record a query result for quality tracking.

        Extracts quality metrics, checks for issues, and updates
        rolling statistics.
        """
        sample = QualitySample(
            timestamp=time.monotonic(),
            query=result.query,
            relevance=result.relevance_score,
            completeness=result.completeness_score,
            coherence=result.coherence_score,
            accuracy=result.accuracy_score,
            overall=result.overall_quality,
            n_blocks=len(result.context_blocks),
            total_tokens=result.total_tokens,
            latency_ms=result.retrieval_time_ms,
        )

        self._samples.append(sample)
        self._total_queries += 1
        self._total_quality_sum += result.overall_quality

        # Detect issues
        self._detect_issues(sample)

    # ──────────────────────────────────────────────────────────────────────
    # Quality Analysis
    # ──────────────────────────────────────────────────────────────────────

    def is_quality_degraded(self) -> bool:
        """Check if quality has recently degraded."""
        if len(self._samples) < 5:
            return False

        recent = self._recent_mean(5)
        baseline = self._recent_mean(min(20, len(self._samples)))

        if baseline < 1e-6:
            return False

        return (baseline - recent) / baseline > self._degradation_threshold

    def is_below_threshold(self) -> bool:
        """Check if latest quality is below alert threshold."""
        if not self._samples:
            return False
        return self._samples[-1].overall < self._alert_threshold

    def get_alert(self) -> Optional[Dict[str, Any]]:
        """
        Get quality alert if triggered.

        Returns alert dict with details, or None if quality is acceptable.
        """
        if not self.is_quality_degraded() and not self.is_below_threshold():
            return None

        latest = self._samples[-1]
        return {
            "type": "degradation" if self.is_quality_degraded() else "below_threshold",
            "current_quality": latest.overall,
            "baseline_quality": self._recent_mean(20) if len(self._samples) >= 20 else self._recent_mean(len(self._samples)),
            "threshold": self._alert_threshold,
            "query": latest.query,
            "timestamp": latest.timestamp,
            "recommendation": "Consider increasing retrieval budget or switching compression strategy.",
        }

    # ──────────────────────────────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive quality statistics."""
        if not self._samples:
            return {
                "total_queries": 0,
                "window_size": self._window_size,
                "samples_in_window": 0,
            }

        overalls = [s.overall for s in self._samples]
        relevances = [s.relevance for s in self._samples]
        latencies = [s.latency_ms for s in self._samples]

        sorted_overalls = sorted(overalls)
        p95_idx = min(int(len(sorted_overalls) * 0.95), len(sorted_overalls) - 1)

        return {
            "total_queries": self._total_queries,
            "window_size": self._window_size,
            "samples_in_window": len(self._samples),
            "mean_quality": self._mean(overalls),
            "min_quality": min(overalls),
            "p95_quality": sorted_overalls[p95_idx],
            "mean_relevance": self._mean(relevances),
            "mean_latency_ms": self._mean(latencies),
            "quality_degraded": self.is_quality_degraded(),
            "recent_issues": len(self._issues[-10:]),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Issue Detection
    # ──────────────────────────────────────────────────────────────────────

    def _detect_issues(self, sample: QualitySample) -> None:
        """Detect quality issues from a sample."""
        if sample.relevance < 0.3:
            self._issues.append(QualityIssue(
                issue_type=QualityIssueType.LOW_RELEVANCE,
                severity=1.0 - sample.relevance,
                description=f"Low relevance ({sample.relevance:.2f}) for query: {sample.query[:50]}",
                suggested_fix="Increase retrieval top_k or improve embeddings.",
            ))

        if sample.overall < self._alert_threshold:
            self._issues.append(QualityIssue(
                issue_type=QualityIssueType.MISSING_INFORMATION,
                severity=1.0 - sample.overall,
                description=f"Overall quality ({sample.overall:.2f}) below threshold.",
                suggested_fix="Consider query refinement or context expansion.",
            ))

        if sample.n_blocks == 0 and sample.total_tokens == 0:
            self._issues.append(QualityIssue(
                issue_type=QualityIssueType.MISSING_INFORMATION,
                severity=1.0,
                description="No context retrieved for query.",
                suggested_fix="Check document store and retrieval configuration.",
            ))

        # Keep only last 100 issues
        if len(self._issues) > 100:
            self._issues = self._issues[-100:]

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _recent_mean(self, n: int) -> float:
        """Mean of last n samples' overall quality."""
        if not self._samples:
            return 0.0
        recent = list(self._samples)[-n:]
        return sum(s.overall for s in recent) / len(recent)

    @staticmethod
    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)
