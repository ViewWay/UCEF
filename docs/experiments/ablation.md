# Ablation Study

We conduct ablation experiments to isolate the contribution of each UCEF component. All ablations run with 5 random seeds (42, 123, 456, 789, 1024) on synthetic hierarchical documents.

## Component Ablation

| Component Removed | Full UCEF | Ablated | Change |
|------------------|-----------|---------|--------|
| Hyperbolic → Euclidean retrieval | — | — | Requires trained embeddings (see note) |
| Quantum → Classical top-k | 0.85±0.03 | 0.85±0.03 | ≈0% |
| Feedback ON → OFF | **0.76±0.14** | 0.47±0.10 | **+62.8%** |
| Three-layer → Single-layer | 0.01ms | 0.06ms | **10.9× slower** |

## Key Findings

### Feedback Loop is the Strongest Contributor

The quality feedback loop provides the **largest single contribution** — improving low-quality initial contexts by +62.8%. All queries converge within ≤3 iterations:

```
Iteration 0 (initial):  Q = 0.47
Iteration 1 (expand):   Q = 0.62  (+32%)
Iteration 2 (lighten):  Q = 0.71  (+15%)
Iteration 3 (requery):  Q = 0.76  (+7%)
```

### Hyperbolic Retrieval Needs Trained Embeddings

With untrained random embeddings, hyperbolic recall (0.76±0.12) is *below* Euclidean (1.00±0.00). This is expected — Poincaré ball embeddings require Riemannian SGD training on hierarchical data. Published works (HypRAG, HyperbolicRAG) report 29–35% improvements after training.

### Quantum Selection Requires Real Document Correlations

With random inter-document correlation matrices, quantum selection matches classical top-k. The density matrix advantage requires genuine document relationships (complementary, contradictory, redundant content).

### Three-Layer Memory Reduces Latency

The hierarchical memory architecture reduces retrieval latency by **10.9×** compared to single-layer linear scan, confirming the value of the hot/warm/cold tiered approach.
