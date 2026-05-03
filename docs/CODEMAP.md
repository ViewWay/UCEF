# UCEF CodeMap

**Version**: 0.3.0 | **Last Updated**: 2026-05-03

---

## Directory Structure

```
extend-Context-System/
├── src/ucef/                          # Main package
│   ├── __init__.py                    # 22 exported symbols ✓
│   ├── core/                          # Core system
│   │   ├── __init__.py                ✓
│   │   ├── system.py                  # UniversalContextSystem ✓
│   │   ├── config.py                  # 9 Pydantic v2 config classes ✓
│   │   └── types.py                   # 17 classes + 32 functions ✓
│   ├── memory/                        # Three-layer memory
│   │   ├── __init__.py                ✓
│   │   ├── hot.py                     # Redis / OrderedDict hot layer ✓
│   │   ├── warm.py                    # ChromaDB / numpy warm layer ✓
│   │   ├── cold.py                    # HDF5 / JSON cold layer ✓
│   │   └── three_layer.py             # Memory orchestrator ✓
│   ├── retrieval/                     # Retrieval methods
│   │   ├── __init__.py                ✓
│   │   ├── hyperbolic.py              # Poincaré ball KNN ✓
│   │   ├── quantum.py                 # Density matrix selection ✓
│   │   ├── fusion.py                  # RRF + weighted fusion ✓
│   │   └── adaptive.py                # Adaptive extension ✓
│   ├── compression/                   # Context compression
│   │   ├── __init__.py                ✓
│   │   ├── adaptive.py                # Strategy router ✓
│   │   ├── mdl.py                     # MDL principle ✓
│   │   ├── entropy.py                 # Max entropy ✓
│   │   └── task_aware.py              # Query-directed ✓
│   ├── physics/                       # Physics-inspired models
│   │   ├── __init__.py                ✓
│   │   ├── thermodynamic.py           # Free energy F=E−TS ✓
│   │   └── quantum_field.py           # Renormalization group ✓
│   ├── quality/                       # Quality preservation
│   │   ├── __init__.py                ✓
│   │   ├── profiler.py                # 12 model profiles ✓
│   │   ├── monitor.py                 # 4-dimension tracking ✓
│   │   ├── feedback.py                # Closed-loop feedback ✓
│   │   └── preservation.py            # Quality engine ✓
│   └── models/                        # Model adapters (lazy loading)
│       ├── __init__.py                ✓
│       ├── base.py                    # Base + AdapterConfig ✓
│       ├── openai.py                  # OpenAI Chat Completions ✓
│       ├── anthropic.py               # Anthropic Messages ✓
│       ├── zhipu.py                   # GLM-4, ChatGLM ✓
│       └── local.py                   # llama.cpp / vLLM / Ollama ✓
│
├── tests/                             # Test suite
│   ├── __init__.py                    ✓
│   ├── run_tests.py                   # Test runner ✓
│   ├── test_config.py                 ✓
│   ├── test_types.py                  ✓
│   ├── test_memory.py                 ✓
│   ├── test_retrieval.py              ✓
│   ├── test_compression.py            ✓
│   ├── test_physics.py                ✓
│   ├── test_quality_models.py         ✓
│   └── test_system_e2e.py             ✓
│
├── experiments/                       # Experiment infrastructure
│   ├── simulated_experiment.py        # 6 simulated experiments ✓
│   ├── real_experiment.py             # Real benchmark runner ✓
│   ├── experiment-report.tex          # TeX experiment report ✓
│   ├── run_all.sh                     # Run all experiments ✓
│   ├── data/                          # Benchmark datasets (synthetic)
│   │   ├── longbench.json             ✓
│   │   ├── narrativeqa.json           ✓
│   │   └── govreport.json             ✓
│   └── results/                       # Experiment outputs
│       ├── simulated_results.json     ✓
│       └── real/                      # Real experiment results
│
├── paper/                             # Research papers
│   ├── ieee/                          # IEEE format
│   │   ├── ucef-en.tex / ucef-en.pdf  ✓
│   │   └── ucef-cn.tex / ucef-cn.pdf  ✓
│   ├── chinese-journal/               # Chinese journal format
│   │   ├── ucef-en.tex / ucef-en.pdf  ✓
│   │   └── ucef-cn.tex / ucef-cn.pdf  ✓
│   └── appendix/                      # Archived drafts
│
├── docs/                              # Documentation
│   ├── PROJECT_SUMMARY.md             ✓
│   ├── CODEMAP.md                     ✓
│   ├── QUICKSTART.md                  ✓
│   ├── RESEARCH_SURVEY.md             ✓
│   └── UPDATE_SUMMARY.md              ✓
│
├── data/cold/                         # Cold memory test data
├── setup.py                           # Package setup (v0.1.0 → needs update)
├── requirements.txt                   # Dependencies
├── pyrightconfig.json                 # Type checker config
└── README.md                          # Project README
```

---

## Module Dependencies

```
ucef/__init__.py
  ├── core/system.py ← core/config.py ← core/types.py
  ├── memory/three_layer.py ← hot.py + warm.py + cold.py
  ├── retrieval/adaptive.py ← hyperbolic.py + quantum.py + fusion.py
  ├── compression/adaptive.py ← mdl.py + entropy.py + task_aware.py
  ├── physics/thermodynamic.py + quantum_field.py
  ├── quality/preservation.py ← monitor.py + feedback.py ← profiler.py
  └── models/base.py → openai.py + anthropic.py + zhipu.py + local.py
```

---

## Public API Surface

### Primary entry points

```python
from ucef import (
    UniversalContextSystem,   # Main orchestrator
    UCEFConfig,               # Configuration
    Document,                 # Input document
    ContextBlock,             # Context unit
    QueryResult,              # Query output
    TokenBudget,              # Token budget
)
```

### Subsystem APIs

```python
# Retrieval
from ucef.retrieval import HyperbolicRetriever, QuantumSelector, AdaptiveContextExtender

# Compression
from ucef.compression import AdaptiveCompressor, MDLCompressor, EntropyCompressor, TaskAwareCompressor

# Quality
from ucef.quality import ModelCapabilityProfiler, QualityMonitor, QualityPreservationEngine

# Memory
from ucef.memory import ThreeLayerMemory, RedisHotMemory, ChromaWarmMemory, FileSystemColdMemory

# Models
from ucef.models import BaseModelAdapter, OpenAIAdapter, AnthropicAdapter, ZhipuAdapter, LocalAdapter
```

---

## Data Flow

```
1. User creates UniversalContextSystem(model_client, model_name, config)
2. system.initialize() → loads profiler, memory, compression, quality engine
3. system.store_documents(docs) →
     docs → blocks → embed in Poincaré ball → store in 3-layer memory
4. system.query(query) →
     profile model → select strategy → retrieve (hyperbolic + quantum)
     → fuse results → compress to budget → quality check → feedback loop
     → return QueryResult
```

---

## Configuration

- **Pydantic v2** with automatic dataclass fallback
- Key configs: HyperbolicConfig, QuantumConfig, CompressionConfig, QualityConfig, MemoryConfig
- All external dependencies optional (graceful degradation)

---

## Testing Strategy

| Module | Test File | Coverage |
|--------|-----------|----------|
| Core types | test_types.py | Types, embeddings, budgets |
| Config | test_config.py | Validation, defaults |
| Memory | test_memory.py | Hot/warm/cold layers |
| Retrieval | test_retrieval.py | Hyperbolic, quantum, fusion |
| Compression | test_compression.py | MDL, entropy, task-aware |
| Physics | test_physics.py | Thermodynamic, RG |
| Quality | test_quality_models.py | Profiler, monitor, feedback |
| Integration | test_system_e2e.py | Full pipeline |

---

## Known Limitations

1. **Hyperbolic retrieval** requires Riemannian SGD training for production-quality embeddings (currently uses random init)
2. **Real benchmarks** require API keys; mock mode produces synthetic results only
3. **CLI interface** defined in setup.py but not yet implemented
4. **Streaming support** has TODO markers in model adapters
5. **setup.py** still reports version 0.1.0 (should be 0.3.0)
