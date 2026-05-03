# UCEF — Universal Context Extension Framework

> **Breaking the Context Barrier**: Model-Agnostic Infinite Context with Quality Preservation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)]()

---

## Overview

UCEF is a universal, model-agnostic framework that enables any LLM to handle unlimited context (4K → 1M+ tokens) while preserving output quality. It combines hyperbolic geometry retrieval, quantum-inspired context selection, adaptive compression, and a three-layer memory architecture into a unified pipeline.

Author: **何红林** (hehonglin525@gmail.com)

---

## Key Features

- **Model-Agnostic** — Works with OpenAI, Anthropic, Zhipu GLM, local models (llama.cpp, vLLM, Ollama)
- **Hyperbolic Retrieval** — Poincaré ball embeddings with geodesic nearest-neighbor search (Ω(log n) over hierarchical data)
- **Quantum-Inspired Selection** — Density matrix formulation for diverse, high-relevance context selection
- **Three-Layer Memory** — Hot (Redis/OrderedDict) / Warm (ChromaDB/numpy) / Cold (HDF5/JSON/Parquet) with automatic tiering
- **Adaptive Compression** — Three strategies: MDL (aggressive ~10%), Entropy (moderate ~30%), Task-Aware (light ~50%)
- **Quality Preservation Engine** — Closed-loop feedback monitoring relevance, completeness, coherence, accuracy
- **Graceful Degradation** — All external dependencies (Redis, ChromaDB, h5py, Pydantic) optional with pure-Python fallbacks

---

## Architecture

```
Query Input
    ↓
┌──────────────────────────────────┐
│  Model Capability Profiler       │
│  Detect context window & strategy│
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│  Multi-Dimensional Retrieval     │
│  ├─ Hyperbolic (Poincaré ball)   │
│  ├─ Quantum (density matrix)     │
│  └─ RRF / Weighted Fusion        │
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│  Adaptive Compression            │
│  MDL / Entropy / Task-Aware      │
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│  Quality Preservation Engine     │
│  Monitor → Feedback → Refine     │
└──────────────┬───────────────────┘
               ↓
        Any LLM → High-Quality Response
```

---

## Simulated Experiment Results

All experiments use synthetic hierarchical documents and mock models. See `experiments/simulated_experiment.py`.

| Metric | Result |
|--------|--------|
| UCEF quality retention (4K→1M) | **89.5%** (vs RAG 71.0%, LongLLMLingua 79.5%) |
| Quantum selection improvement | +1.9% accuracy over classical top-k |
| Aggressive compression retention | ~10% size, ~92.6% quality |
| Moderate compression retention | ~28% size, ~72.0% quality |
| Light compression retention | ~50% size, ~69.0% quality |
| Feedback loop convergence (≤3 iters) | **100%** |
| Total pipeline latency | **~24ms** (target: <500ms) |

Note: Hyperbolic retrieval with random (untrained) embeddings underperforms TF-IDF, as expected — the hierarchical advantage requires Riemannian SGD training.

---

## Quick Start

### Installation

```bash
git clone https://github.com/ViewWay/UCEF.git
cd extend-Context-System
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Basic Usage

```python
import asyncio
from ucef import UniversalContextSystem, UCEFConfig, Document

async def main():
    system = UniversalContextSystem(
        model_client=your_model_client,
        model_name="your-model",
    )
    await system.initialize()

    # Store large documents
    docs = [Document(id="doc1", text="..."), Document(id="doc2", text="...")]
    await system.store_documents(docs)

    # Query with automatic context extension
    result = await system.query("Analyze the project architecture")
    print(result.format_context())

asyncio.run(main())
```

### Running Experiments

```bash
# Simulated experiments (no API key needed)
python experiments/simulated_experiment.py

# Real experiments (set API keys first)
export OPENAI_API_KEY=sk-...
python experiments/real_experiment.py -b all -m gpt-4o -n 100
```

### Running Tests

```bash
pytest tests/
```

---

## Project Structure

```
extend-Context-System/
├── src/ucef/                    # Main package
│   ├── __init__.py              # 22 exported symbols
│   ├── core/                    # Core system
│   │   ├── system.py            # UniversalContextSystem orchestrator
│   │   ├── config.py            # Pydantic v2 configuration (9 classes)
│   │   └── types.py             # Mathematical types (17 classes, 32 functions)
│   ├── memory/                  # Three-layer memory
│   │   ├── hot.py               # Redis / OrderedDict hot layer
│   │   ├── warm.py              # ChromaDB / numpy warm layer
│   │   ├── cold.py              # HDF5 / JSON cold layer
│   │   └── three_layer.py       # Memory orchestrator
│   ├── retrieval/               # Retrieval methods
│   │   ├── hyperbolic.py        # Poincaré ball embeddings
│   │   ├── quantum.py           # Quantum-inspired selection
│   │   ├── fusion.py            # RRF + weighted fusion
│   │   └── adaptive.py          # Adaptive context extension
│   ├── compression/             # Context compression
│   │   ├── adaptive.py          # Strategy router
│   │   ├── mdl.py               # MDL principle
│   │   ├── entropy.py           # Max entropy selection
│   │   └── task_aware.py        # Query-directed extraction
│   ├── physics/                 # Physics-inspired models
│   │   ├── thermodynamic.py     # Free energy minimization
│   │   └── quantum_field.py     # Renormalization group
│   ├── quality/                 # Quality preservation
│   │   ├── profiler.py          # Model capability profiling (12 models)
│   │   ├── monitor.py           # Real-time quality tracking
│   │   ├── feedback.py          # Closed-loop feedback
│   │   └── preservation.py      # Quality engine
│   └── models/                  # Model adapters (lazy loading)
│       ├── base.py              # Base adapter
│       ├── openai.py            # OpenAI adapter
│       ├── anthropic.py         # Anthropic adapter
│       ├── zhipu.py             # Zhipu GLM adapter
│       └── local.py             # Local model adapter
├── tests/                       # Test suite (8 test modules)
├── experiments/                 # Simulated + real experiments
├── paper/                       # Research papers
│   ├── ieee/                    # IEEE format (EN + CN)
│   ├── chinese-journal/         # Chinese journal format (EN + CN)
│   └── appendix/                # Archived drafts
├── docs/                        # Documentation
├── setup.py                     # Package setup
└── requirements.txt             # Dependencies
```

---

## Supported Models

| Model | Provider | Context Window |
|-------|----------|---------------|
| GPT-4o | OpenAI | 128K |
| Claude 3.5 Sonnet | Anthropic | 200K |
| GLM-4 | Zhipu AI | 128K |
| GLM-4-Long | Zhipu AI | 1M |
| Llama (local) | llama.cpp / vLLM / Ollama | varies |

The profiler supports 12 pre-configured model profiles covering small (4K–8K), medium (32K–64K), and large (128K+) context windows.

---

## Supported Benchmarks

| Benchmark | Task | Description |
|-----------|------|-------------|
| LongBench | Multi-task | Long context QA, summarization, retrieval, classification |
| NarrativeQA | Document QA | Narrative document question answering |
| GovReport | Summarization | Government report summarization |

---

## Research Contributions

1. **UCEF** — Universal Context Extension Framework, first model-agnostic framework for LLM context extension
2. **TSR** — Topological Semantic Retrieval via hyperbolic space (Ω(log n) guarantee)
3. **QCS** — Quantum Context Selection via density matrix formulation
4. **AES** — Adaptive Extension Strategy for 4K–200K → 1M+
5. **QPE** — Quality Preservation Engine with closed-loop feedback
6. **3LMA** — Three-Layer Memory Architecture with tiered latency guarantees
7. **ABA** — Adaptive Budget Allocation via knapsack optimization

---

## Citation

```bibtex
@article{he2026ucef,
  title={Breaking the Context Barrier: A Universal Context Extension Framework for LLMs},
  author={何红林},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT License

---

**Repository**: https://github.com/ViewWay/UCEF
