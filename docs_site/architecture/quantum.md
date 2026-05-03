# Quantum-Inspired Selection

Context selection uses the mathematical framework of quantum probability theory. Documents are quantum states, and the query acts as a measurement operator.

## Mathematical Foundation

### Superposition

All candidate documents form a quantum superposition:

$$|\psi\rangle = \sum_i \sqrt{p_i} \, |\text{doc}_i\rangle$$

where $p_i$ are relevance-based probabilities (Born rule: $P(i) = |\alpha_i|^2 = p_i$).

### Density Matrix

$$\rho = |\psi\rangle\langle\psi| + \text{entanglement corrections}$$

The off-diagonal elements capture document-document correlations (entanglement).

### Measurement

Query as measurement operator:

$$S(q, c) = \langle q|\rho|c\rangle$$

This produces measurement probabilities that combine classical relevance with quantum correlations.

### Interference

$$I(i) = \left|\sum_j \alpha_i \alpha_j^* \cos(\theta_{ij})\right|^2$$

Constructive interference boosts coherent documents, destructive interference suppresses contradictions.

## Pipeline

```
1. Build superposition |ψ⟩ from relevance scores
2. Construct density matrix ρ = |ψ⟩⟨ψ| + entanglement
3. Apply query measurement → measurement probabilities
4. Apply interference filtering (optional)
5. Collapse to top-k contexts respecting budget
```

## API

```python
from ucef.retrieval.quantum import QuantumSelector
from ucef.core.config import QuantumConfig

config = QuantumConfig(
    enabled=True,
    initial_amplitude="relevance_weighted",
    measurement_method="argmax",
    use_interference=True,
)

selector = QuantumSelector(config)
blocks = selector.select(scored_documents, query_weights, budget)
```

## Measurement Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `argmax` | Select highest probability | Deterministic, reproducible |
| `top_k` | Select top-k by probability | Controlled diversity |
| `sampling` | Probabilistic sampling | Exploration, diversity |

## Experiment Results

- **+1.9%** accuracy improvement over classical top-k
- **3.00** focused topics (vs 7.00 scattered with classical)

## Reference

van Rijsbergen, "The Geometry of Information Retrieval", Cambridge University Press, 2004.
