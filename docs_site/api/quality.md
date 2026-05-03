# Quality API

## QualityFeedbackLoop

```python
from ucef.quality.feedback import QualityFeedbackLoop, FeedbackResult

loop = QualityFeedbackLoop(
    max_iterations=3,
    target_quality=0.75,
    min_improvement=0.02,
    initial_top_k=10,
    top_k_multiplier=1.5,
)

feedback = await loop.refine(
    initial_result=result,
    requery_fn=system.query,
    query="user query",
    quality_threshold=0.8,
)

feedback.final_result           # Best QueryResult
feedback.iterations             # Number of iterations
feedback.converged              # Whether quality threshold met
feedback.total_improvement      # Quality delta
feedback.total_refinement_ms    # Time spent
feedback.steps                  # List[RefinementStep]
```

## QualityMonitor

```python
from ucef.quality.monitor import QualityMonitor

monitor = QualityMonitor(
    window_size=100,
    alert_threshold=0.7,
)

monitor.record(query_result)

stats = monitor.get_stats()
# → {"total_queries": N, "mean_quality": ..., "alerts": ...}
```

## ModelCapabilityProfiler

```python
from ucef.quality.profiler import ModelCapabilityProfiler

profiler = ModelCapabilityProfiler()
profile = await profiler.profile_model(client, "gpt-4o")

profile.native_context_window    # 128000
profile.quality_retention        # 0.94
profile.recommended_strategy     # CompressionStrategy.LIGHT
profile.classify_context_category()  # ContextCategory.LARGE
```

## QualityPreservationEngine

```python
from ucef.quality.preservation import QualityPreservationEngine

engine = QualityPreservationEngine(
    monitor=monitor,
    feedback_loop=loop,
)

result = await engine.preserve_quality(
    query_result=result,
    threshold=0.75,
)
```
