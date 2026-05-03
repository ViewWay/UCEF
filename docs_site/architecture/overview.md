# Architecture Overview

UCEF is organized as a pipeline of modular subsystems, each grounded in mathematical theory.

## System Pipeline

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

## Module Structure

| Module | Files | Purpose |
|--------|-------|---------|
| `core/` | types.py, config.py, system.py | Mathematical types, configuration, orchestrator |
| `memory/` | hot.py, warm.py, cold.py, three_layer.py | Three-layer storage |
| `retrieval/` | hyperbolic.py, quantum.py, fusion.py, adaptive.py | Context retrieval |
| `compression/` | mdl.py, entropy.py, task_aware.py, adaptive.py | Context compression |
| `physics/` | thermodynamic.py, quantum_field.py | Physics-inspired models |
| `quality/` | profiler.py, monitor.py, feedback.py, preservation.py | Quality preservation |
| `models/` | base.py, openai.py, anthropic.py, zhipu.py, local.py | Model adapters |

## Key Design Principles

1. **Model-Agnostic** — All strategies adapt to model capabilities via `ModelProfile`
2. **Graceful Degradation** — External deps optional, pure-Python fallbacks
3. **Async-First** — All I/O operations are async
4. **Lazy Loading** — Model adapters imported only when needed
5. **Closed-Loop Quality** — Feedback loop ensures quality targets are met

## Data Flow

```
1. UniversalContextSystem.initialize()
   → Profile model → Create budget → Setup subsystems

2. store_documents(docs)
   → Tokenize → Embed → Distribute to memory layers

3. query(query)
   → Retrieve candidates → Score → Quantum select
   → Compress to budget → Evaluate quality → Feedback loop
   → Return QueryResult
```
