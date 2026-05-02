# Quality System API Reference

## Overview

The Quality System ensures output quality is maintained or improved when using context extension.

## Modules

### 1. ModelCapabilityProfiler

**Location**: `src/ucef/quality/profiler.py`

Analyzes model characteristics to select optimal context extension strategy.

#### Usage

```python
from ucef.quality import ModelCapabilityProfiler

profiler = ModelCapabilityProfiler()
profile = await profiler.profile_model(model_client, "llama-7b")

print(f"Native context: {profile.native_context_window}")
print(f"Category: {profile.context_category}")
print(f"Recommended strategy: {profile.recommended_strategy}")
```

#### Attributes

- `MODEL_SPECS` (dict): Known model specifications
- `cache` (dict): Cached model profiles

#### Methods

**`async profile_model(model_client, model_name)`**
- Returns: `ModelProfile`
- Purpose: Analyze model capabilities

**`_get_context_spec(model_name)`**
- Returns: `(context_window, category)`
- Purpose: Get context window from model name

---

### 2. QualityPreservationEngine

**Location**: `src/ucef/quality/preservation.py`

Ensures output quality meets threshold through monitoring and refinement.

#### Usage

```python
from ucef.quality import QualityPreservationEngine

engine = QualityPreservationEngine(quality_threshold=0.75)

# Ensure quality
response = await engine.ensure_quality(
    model_client,
    query="What is the project architecture?",
    current_context=selected_docs,
    initial_response=initial_response
)
```

#### Methods

**`async ensure_quality(model_client, query, current_context, initial_response)`**
- Returns: `str` (high-quality response)
- Purpose: Ensure output meets quality threshold

**`async _evaluate_response_quality(response, query, context)`**
- Returns: `float` (0-1 score)
- Purpose: Evaluate response quality

**`async _identify_quality_issues(response, query, context)`**
- Returns: `List[QualityIssue]`
- Purpose: Identify specific quality problems

---

### 3. AdaptiveContextExtender

**Location**: `src/ucef/retrieval/adaptive.py`

Automatically selects optimal extension strategy based on model profile.

#### Usage

```python
from ucef.retrieval import AdaptiveContextExtender

extender = AdaptiveContextExtender()
extended_context = await extender.extend_context(
    model_profile=profile,
    documents=all_documents,
    query="user query"
)
```

#### Strategies

**SmallContextStrategy** (4K-32K)
- Aggressive compression (10%)
- Precision retrieval
- Multi-round fusion

**MediumContextStrategy** (32K-128K)
- Moderate compression (30%)
- Hyperbolic retrieval
- Quantum selection

**LargeContextStrategy** (128K-200K)
- Light compression (50%)
- Structure preservation
- Attention optimization

---

## Data Classes

### ModelProfile

```python
@dataclass
class ModelProfile:
    model_name: str
    native_context_window: int
    context_category: str
    performance_curve: Dict[int, float]
    quality_retention: float
    retrieval_capability: float
    reasoning_strength: float
    recommended_strategy: str
    optimal_compression_ratio: float
    max_extended_context: int
```

### QualityIssue

```python
@dataclass
class QualityIssue:
    type: str
    severity: float
    description: str
    suggested_fix: str
```

---

## Examples

### Example 1: Profile a Model

```python
from ucef.quality import ModelCapabilityProfiler

async def main():
    profiler = ModelCapabilityProfiler()
    
    # Profile different models
    for model_name in ["llama-7b", "llama-13b", "gpt-4o"]:
        profile = await profiler.profile_model(None, model_name)
        print(f"{model_name}:")
        print(f"  Context: {profile.native_context_window}")
        print(f"  Strategy: {profile.recommended_strategy}")
        print(f"  Max Extended: {profile.max_extended_context}")
        print()
```

### Example 2: Ensure Quality

```python
from ucef.quality import QualityPreservationEngine

async def query_with_quality_guarantee(system, query):
    engine = QualityPreservationEngine(threshold=0.80)
    
    # Get initial response
    initial = await system.model.generate(query)
    
    # Ensure quality
    final = await engine.ensure_quality(
        system.model,
        query,
        system.current_context,
        initial
    )
    
    return final
```

### Example 3: Adaptive Extension

```python
from ucef.retrieval import AdaptiveContextExtender
from ucef.quality import ModelCapabilityProfiler

async def smart_context_extension(model, docs, query):
    # 1. Profile model
    profiler = ModelCapabilityProfiler()
    profile = await profiler.profile_model(model, "llama-7b")
    
    # 2. Extend with adaptive strategy
    extender = AdaptiveContextExtender()
    context = await extender.extend_context(
        profile,
        docs,
        query
    )
    
    # 3. Use context
    return context
```

---

## Quality Metrics

The quality system evaluates responses on 4 dimensions:

1. **Relevance** (30%): How well response addresses query
2. **Completeness** (30%): How thoroughly response answers
3. **Coherence** (20%): Logical flow and structure
4. **Accuracy** (20%): Factual correctness

Overall quality = weighted sum of these metrics.

---

## Configuration

### Quality Thresholds

- `quality_threshold`: Minimum acceptable quality (default: 0.75)
- Can be adjusted per model or use case

### Strategy Selection

Automatic based on:
- Native context window size
- Quality retention capability
- Reasoning strength

---

## Performance

### Profiling Overhead
- Cached models: <1ms
- Unknown models: ~100ms (one-time)

### Quality Checking
- Evaluation: ~50ms
- Refinement (if needed): +200ms

### Total Overhead
- With good quality: +50ms
- With refinement: +250ms

---

## Error Handling

### Common Issues

1. **Unknown Model**
   - Fallback to medium strategy
   - Estimate capabilities from size

2. **Quality Below Threshold**
   - Automatic refinement
   - Context re-selection
   - Regeneration

3. **Profile Cache Miss**
   - Automatic profiling
   - Cache for future use
