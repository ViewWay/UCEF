# UCEF Project Summary

**Version**: 0.3.0
**Author**: 何红林 (hehonglin525@gmail.com)
**Repository**: https://github.com/ViewWay/UCEF
**Status**: Research prototype — all core modules implemented, simulated experiments validated

---

## What is UCEF

Universal Context Extension Framework (UCEF) enables any LLM to handle unlimited context (4K → 1M+ tokens) while preserving output quality. It is model-agnostic and works with OpenAI, Anthropic, Zhipu, and local models.

---

## Implementation Status

All six development phases are complete:

| Phase | Scope | Status | Key Files |
|-------|-------|--------|-----------|
| 1. Core System | Types, config, orchestrator | ✅ | core/types.py, core/config.py, core/system.py |
| 2. Retrieval & Memory | Hyperbolic, quantum, 3-layer | ✅ | retrieval/*.py, memory/*.py |
| 3. Compression & Physics | MDL, entropy, task-aware, thermodynamic, RG | ✅ | compression/*.py, physics/*.py |
| 4. Quality System | Profiler, monitor, feedback, preservation | ✅ | quality/*.py |
| 5. Model Adapters | OpenAI, Anthropic, Zhipu, local | ✅ | models/*.py |
| 6. Testing & Experiments | 8 test modules, simulated + real experiments | ✅ | tests/*.py, experiments/*.py |

---

## Module Summary

### core/ — Foundation
- **types.py** — 17 classes + 32 functions for hyperbolic geometry, quantum states, information-theoretic types
- **config.py** — 9 Pydantic v2 configuration classes with dataclass fallback
- **system.py** — UniversalContextSystem orchestrator coordinating all subsystems

### memory/ — Three-Layer Architecture
- **hot.py** — Redis / OrderedDict hot layer, LRU + TTL eviction, <10ms target
- **warm.py** — ChromaDB / numpy warm layer, semantic retrieval, <100ms target
- **cold.py** — HDF5 / JSON / Parquet cold layer, unlimited capacity, <500ms target
- **three_layer.py** — Orchestrator coordinating tiering and migration policies

### retrieval/ — Multi-Dimensional Retrieval
- **hyperbolic.py** — Poincaré ball embeddings with geodesic KNN (Ω(log n))
- **quantum.py** — Density matrix selection with superposition and measurement
- **fusion.py** — Reciprocal Rank Fusion + weighted score fusion
- **adaptive.py** — Strategy selection based on model capability profile

### compression/ — Adaptive Compression
- **mdl.py** — Minimum Description Length principle
- **entropy.py** — Maximum entropy for diversity-preserving selection
- **task_aware.py** — Query-directed sentence extraction (~60% compression, ~92% quality)
- **adaptive.py** — Strategy router: aggressive (10%), moderate (30%), light (50%)

### physics/ — Physics-Inspired Models
- **thermodynamic.py** — Free energy minimization (F = E − T·S)
- **quantum_field.py** — Wilson's renormalization group for multi-scale coarse-graining

### quality/ — Quality Preservation
- **profiler.py** — Model capability profiling for 12 models
- **monitor.py** — Real-time quality tracking across 4 dimensions
- **feedback.py** — Closed-loop feedback with expand/relax/re-compress actions
- **preservation.py** — Quality engine ensuring output quality maintained or improved

### models/ — Model Adapters
- **base.py** — Common functionality: token counting, error handling, retry
- **openai.py** — OpenAI Chat Completions with tiktoken + exponential backoff
- **anthropic.py** — Anthropic Messages API
- **zhipu.py** — Zhipu GLM-4, ChatGLM
- **local.py** — llama.cpp, vLLM, Ollama local inference

---

## Experiment Results (Simulated)

| Experiment | Key Result |
|-----------|------------|
| Hyperbolic vs Euclidean retrieval | Below baseline with random embeddings (needs Riemannian SGD training) |
| Quantum vs classical selection | +1.9% accuracy, more focused topic coverage |
| Compression strategy comparison | Aggressive 10%/92.6%, Moderate 28%/72%, Light 50%/69% |
| End-to-end quality (4K→1M) | UCEF: 89.5% vs RAG: 71.0% vs LongLLMLingua: 79.5% |
| Feedback loop convergence | 100% within ≤3 iterations |
| Pipeline latency | ~24ms average (target <500ms) |

---

## Papers

Four versions compiled and available in `paper/`:

- `paper/ieee/ucef-en.pdf` — IEEE English (4 pages)
- `paper/ieee/ucef-cn.pdf` — IEEE Chinese (5 pages)
- `paper/chinese-journal/ucef-en.pdf` — Chinese journal English (10 pages)
- `paper/chinese-journal/ucef-cn.pdf` — Chinese journal Chinese (8 pages)

Experiment report: `experiments/experiment-report.tex`

---

## Test Suite

8 test modules in `tests/`:

- test_config.py — Configuration validation
- test_types.py — Mathematical type system
- test_memory.py — Three-layer memory
- test_retrieval.py — Hyperbolic and quantum retrieval
- test_compression.py — Compression algorithms
- test_physics.py — Physics-inspired models
- test_quality_models.py — Quality preservation
- test_system_e2e.py — End-to-end integration

---

## Real Experiment Infrastructure

`experiments/real_experiment.py` supports:

- **Benchmarks**: LongBench, NarrativeQA, GovReport
- **Models**: GPT-4o, Claude 3.5 Sonnet, GLM-4, GLM-4-Long
- **Metrics**: ROUGE-L, approximate BERTScore, Recall@K
- **Fallback**: Mock mode when no API keys available

Usage: `python experiments/real_experiment.py -b all -m gpt-4o -n 100`

---

**Last Updated**: 2026-05-03
