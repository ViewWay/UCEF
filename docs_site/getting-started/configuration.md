# Configuration

UCEF uses Pydantic v2 configuration classes with automatic dataclass fallback.

## Basic Config

```python
from ucef import UCEFConfig

# Default configuration
config = UCEFConfig()

# Customize
config.hyperbolic.embedding_dim = 16
config.quantum.enabled = True
config.compression.aggressive_ratio = 0.1
config.quality.quality_threshold = 0.75
```

## Hyperbolic Retrieval

```python
from ucef.core.config import HyperbolicConfig

config = HyperbolicConfig(
    embedding_dim=8,       # Poincaré ball dimension
    max_norm=0.9,          # Maximum embedding norm (< 1.0)
    n_neighbors=50,        # KNN neighbors
    curvature=-1.0,        # Hyperbolic curvature
)
```

## Quantum Selection

```python
from ucef.core.config import QuantumConfig

config = QuantumConfig(
    enabled=True,                     # Enable quantum selection
    initial_amplitude="relevance_weighted",  # or "equal", "entropy_weighted"
    measurement_method="argmax",      # or "sampling", "top_k"
    top_k_measurements=10,
    use_interference=True,
    entanglement_threshold=0.3,
)
```

## Compression

```python
from ucef.core.config import CompressionConfig

config = CompressionConfig(
    aggressive_ratio=0.10,            # MDL: retain 10%
    moderate_ratio=0.30,              # Entropy: retain 30%
    description_length_weight=0.5,    # MDL balance
    use_mdl=True,
    use_entropy=True,
)
```

## Quality

```python
from ucef.core.config import QualityConfig

config = QualityConfig(
    quality_threshold=0.75,           # Minimum quality score
    max_refinement_iterations=3,      # Feedback loop limit
    monitor_window_size=100,          # Quality tracking window
)
```

## Model Profile

UCEF automatically profiles the model and selects optimal strategies:

```python
from ucef.quality.profiler import ModelCapabilityProfiler

profiler = ModelCapabilityProfiler()
profile = await profiler.profile_model(model_client, "gpt-4o")

print(f"Context window: {profile.native_context_window}")
print(f"Category: {profile.classify_context_category()}")
print(f"Strategy: {profile.recommended_strategy}")
print(f"Quality retention: {profile.quality_retention}")
```

## Graceful Degradation

UCEF handles missing dependencies automatically:

- **Pydantic not installed** → falls back to dataclasses
- **Redis not installed** → uses OrderedDict for hot memory
- **ChromaDB not installed** → uses numpy for warm memory
- **h5py not installed** → uses JSON for cold memory
- **No API keys** → mock mode for experiments
