# Quality Preservation

Closed-loop feedback system ensuring output quality is maintained or improved across all context extension scenarios.

## Quality Model

$$Q = 0.30 \cdot \text{Relevance} + 0.30 \cdot \text{Completeness} + 0.20 \cdot \text{Coherence} + 0.20 \cdot \text{Accuracy}$$

| Dimension | Measurement | Weight |
|-----------|------------|--------|
| Relevance | Block-level relevance scores | 0.30 |
| Completeness | Query term coverage in context | 0.30 |
| Coherence | Block count and diversity | 0.20 |
| Accuracy | Relevance × completeness × coherence heuristic | 0.20 |

## Feedback Loop

When quality is below threshold, iteratively applies refinement:

```
1. EXPAND_RETRIEVAL   → Increase top_k (×1.5)
2. LIGHTEN_COMPRESSION → Switch to lighter strategy
3. FULL_REQUERY       → Restart with expanded parameters
```

Maximum 3 iterations by default. Convergence criteria:

- Quality ≥ threshold → stop
- Improvement < 0.02 → stop (stagnation)
- Max iterations reached → return best result

## Experiment Results

At threshold 0.75:

| Iterations | Percentage |
|-----------|-----------|
| 1 | 10.0% |
| 2 | 40.0% |
| 3 | 50.0% |
| **≤3 total** | **100%** |

## API

```python
from ucef.quality.feedback import QualityFeedbackLoop, FeedbackResult

loop = QualityFeedbackLoop(
    max_iterations=3,
    target_quality=0.75,
    min_improvement=0.02,
)

feedback = await loop.refine(
    initial_result=result,
    requery_fn=system.query,
    query="user query",
)

print(f"Converged: {feedback.converged}")
print(f"Iterations: {feedback.iterations}")
print(f"Improvement: {feedback.total_improvement:.2%}")
```

## Quality Monitor

Real-time quality tracking with configurable alert threshold:

```python
from ucef.quality.monitor import QualityMonitor

monitor = QualityMonitor(
    window_size=100,
    alert_threshold=0.7,
)

# Record results
monitor.record(query_result)

# Get statistics
stats = monitor.get_stats()
# → mean_quality, min_quality, alert_count, etc.
```

## Model Profiling

```python
from ucef.quality.profiler import ModelCapabilityProfiler

profiler = ModelCapabilityProfiler()
profile = await profiler.profile_model(model_client, "gpt-4o")

# Access profile
profile.native_context_window   # 128000
profile.quality_retention       # 0.94
profile.recommended_strategy    # CompressionStrategy.LIGHT
profile.classify_context_category()  # ContextCategory.LARGE
```

Supports 12 pre-configured model profiles.
