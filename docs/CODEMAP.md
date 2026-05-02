# UCEF CodeMap

## Overview
UCEF (Universal Context Extension Framework) - Model-agnostic infinite context with quality preservation.

## Directory Structure

```
extend-Context-System/
├── src/ucef/                      # Main package
│   ├── core/                      # Core system
│   │   ├── __init__.py
│   │   ├── system.py              # Main UCEF system (TODO)
│   │   └── config.py              # Configuration (TODO)
│   ├── memory/                    # Memory systems
│   │   ├── __init__.py
│   │   ├── hot.py                 # Redis hot memory (TODO)
│   │   ├── warm.py               # ChromaDB warm memory (TODO)
│   │   ├── cold.py               # File system cold memory (TODO)
│   │   └── three_layer.py        # 3-layer architecture (TODO)
│   ├── retrieval/                 # Retrieval methods
│   │   ├── __init__.py
│   │   ├── hyperbolic.py         # Hyperbolic retrieval (TODO)
│   │   ├── quantum.py            # Quantum selection (TODO)
│   │   └── adaptive.py           # Adaptive strategies ✓
│   ├── models/                    # Model adapters
│   │   ├── __init__.py
│   │   ├── base.py               # Base adapter (TODO)
│   │   ├── openai.py             # OpenAI adapter (TODO)
│   │   ├── anthropic.py          # Anthropic adapter (TODO)
│   │   ├── zhipu.py              # Zhipu GLM adapter (TODO)
│   │   └── local.py              # Local model adapter (TODO)
│   ├── quality/                   # Quality preservation
│   │   ├── __init__.py
│   │   ├── profiler.py           # Model profiler ✓
│   │   ├── preservation.py       # Quality engine ✓
│   │   └── monitor.py            # Quality monitor (TODO)
│   ├── adapters/                  # Additional adapters (empty)
│   └── utils.py                   # Utilities (TODO)
├── tests/                         # Tests
│   ├── unit/                      # Unit tests
│   └── integration/               # Integration tests
├── paper/                         # Research paper
│   ├── draft/                     # Paper drafts
│   ├── figures/                   # Figures
│   └── appendix/                  # Supplementary
├── experiments/                   # Experiments
│   ├── benchmarks/                # Benchmark datasets
│   ├── baselines/                 # Baseline implementations
│   └── results/                   # Experiment results
├── docs/                          # Documentation
│   ├── api/                       # API reference
│   ├── tutorials/                 # Tutorials
│   └── theory/                    # Theoretical background
├── scripts/                       # Utility scripts
│   ├── setup/                     # Setup scripts
│   ├── eval/                      # Evaluation scripts
│   └── plot/                      # Plotting scripts
├── examples/                      # Usage examples
│   └── basic_usage.py             # Basic examples ✓
└── data/                          # Data storage
    ├── cache/                     # Cached data
    └── models/                    # Downloaded models
```

## Key Components

### ✓ Implemented
- **Project structure** - Complete directory layout
- **Quality profiler** - Model capability analysis
- **Quality preservation** - Output quality engine
- **Adaptive strategies** - Context extension strategies
- **Documentation** - Comprehensive docs and examples

### ⏳ TODO
- Core system integration
- Memory system implementation
- Retrieval methods (hyperbolic, quantum)
- Model adapters (OpenAI, Anthropic, etc.)
- Testing framework
- Experiment scripts

## Module Dependencies

### Quality System
```
profiler.py (standalone)
  ↓
preservation.py (standalone)
  ↓
adaptive.py (uses both)
```

### Planned Integration
```
system.py
  ├── profiler.py
  ├── adaptive.py
  ├── memory/
  ├── retrieval/
  └── models/
```

## API Surface

### Public API
```python
from ucef import (
    UniversalContextSystem,
    ModelCapabilityProfiler,
    AdaptiveContextExtender
)
```

### Internal API
```python
from ucef.quality import ModelCapabilityProfiler, QualityPreservationEngine
from ucef.retrieval import AdaptiveContextExtender
from ucef.memory import ThreeLayerMemory
from ucef.models import BaseModelAdapter, OpenAIAdapter, etc.
```

## Data Flow

### Input
```python
model_client = load_model("llama-7b")
documents = load_documents()  # 1M+ tokens
query = "user query"
```

### Processing
```python
profile = profiler.profile(model_client, "llama-7b")
context = extender.extend(profile, documents, query)
response = quality_engine.ensure(model_client, query, context)
```

### Output
```python
high_quality_response  # Comparable to GPT-4o
```

## Entry Points

### Main Entry
- `src/ucef/__init__.py` - Package initialization

### Quality Entry
- `src/ucef/quality/profiler.py` - Model profiling
- `src/ucef/quality/preservation.py` - Quality preservation

### Retrieval Entry
- `src/ucef/retrieval/adaptive.py` - Adaptive extension

## Configuration

### Model Specifications
Located in `profiler.py`:
- `MODEL_SPECS` dict
- 15+ pre-configured models
- Categories: small/medium/large

### Quality Thresholds
Located in `preservation.py`:
- Default: 0.75
- Configurable per model

### Strategy Selection
Located in `adaptive.py`:
- Automatic based on profile
- 3 strategies implemented

## Testing Strategy

### Unit Tests
- Each module independently
- Mock external dependencies
- Fast execution

### Integration Tests
- End-to-end workflows
- Real model clients
- Performance benchmarks

## Documentation

### API Docs
- `docs/api/quality-system.md` - Quality system API
- `docs/api/architecture.md` - Architecture

### User Docs
- `docs/QUICKSTART.md` - Quick start
- `examples/basic_usage.py` - Usage examples

### Developer Docs
- `docs/PROJECT_SUMMARY.md` - Project overview
- `docs/research-plan.md` - Research plan

## Dependencies

### Core Dependencies
- numpy, scipy, torch
- chromadb
- redis
- openai, anthropic, zhipuai

### Development Dependencies
- pytest, pytest-cov
- black, flake8, mypy

## Performance Targets

### Profiling
- Cached: <1ms
- Uncached: ~100ms

### Extension
- Small strategy: ~200ms
- Medium strategy: ~150ms
- Large strategy: ~100ms

### Quality Check
- Evaluation: ~50ms
- Refinement: +200ms (if needed)

## Quality Metrics

### 4 Dimensions
1. Relevance (30%)
2. Completeness (30%)
3. Coherence (20%)
4. Accuracy (20%)

### Threshold
- Default: 0.75
- Adjustable per model

## Model Support

### Currently Supported (15+)
Small: Llama-7B, Qwen-7B, Mistral-7B
Medium: Llama-13B, Qwen-14B, Yi-34B
Large: Llama-70B, Qwen2.5-72B, GLM-5.1, GPT-4o, Claude 3.5

### Extensibility
- Easy to add new models
- Update MODEL_SPECS dict
- Or provide custom profile

## File Status Legend
- ✓ Implemented
- ⏳ TODO
- 🔄 In progress
- ❌ Deprecated

---

**Last Updated**: 2026-05-02
**Version**: 0.1.0
**Status**: Framework Complete, Implementation In Progress
