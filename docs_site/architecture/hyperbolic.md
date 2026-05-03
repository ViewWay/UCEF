# Hyperbolic Retrieval

Documents are embedded as points in the Poincaré ball model of hyperbolic space. Retrieval uses geodesic distance, which naturally captures hierarchical semantic relationships.

## Mathematical Foundation

### Poincaré Ball Model

The Poincaré ball $\mathbb{B}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$ with Riemannian metric:

$$g_x = \lambda_x^2 \cdot I, \quad \lambda_x = \frac{2}{1 - \|x\|^2}$$

### Geodesic Distance

$$d(u, v) = \text{arcosh}\left(1 + \frac{2\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}\right)$$

This distance has the key property that points near the boundary are exponentially far apart, naturally encoding hierarchies.

### Möbius Addition

$$u \oplus v = \frac{(1 + 2\langle u, v\rangle + \|v\|^2)u + (1 - \|u\|^2)v}{1 + 2\langle u, v\rangle + \|u\|^2\|v\|^2}$$

### Exponential Map

At the origin: $\exp_0(v) = \tanh(\|v\|) \cdot \frac{v}{\|v\|}$

General case: $\exp_{\text{base}}(v) = \text{base} \oplus \tanh\left(\frac{\lambda_{\text{base}} \cdot \|v\|}{2}\right) \cdot \frac{v}{\|v\|}$

## API

```python
from ucef.retrieval.hyperbolic import HyperbolicRetriever
from ucef.core.types import Document, HyperbolicPoint

# Create and index
retriever = HyperbolicRetriever()
retriever.index(documents)

# Retrieve by point
query_point = HyperbolicPoint.random(dim=8)
results = retriever.retrieve(query_point, top_k=10)

# Retrieve by text (auto-embeds)
results = retriever.retrieve_by_text("machine learning", top_k=10)

for doc, distance in results:
    print(f"{doc.id}: distance={distance:.4f}")
```

## Complexity

- **Indexing**: O(n · d) — compute/store embeddings
- **Retrieval**: O(n · d) — vectorized distance computation
- **With HNSW index**: O(log n · d) — future optimization

## Reference

Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations", NeurIPS 2017.
