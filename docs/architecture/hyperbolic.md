---
title: Hyperbolic Geometry
---

# Hyperbolic Geometry — Poincare Ball Retrieval

UCEF's retrieval engine embeds documents as points in hyperbolic space using the Poincare ball model. This page provides the mathematical foundations, algorithm details, and implementation considerations.

---

## Why Hyperbolic Space?

Standard Euclidean embeddings struggle with hierarchical data. A tree with branching factor $b$ and depth $d$ requires $\Theta(b^d)$ volume to represent in Euclidean space, but only $\Theta(d)$ distance from root to leaf in hyperbolic space.

The key insight from Nickel & Kiela (2017): hyperbolic space has **exponentially expanding volume** near the boundary, making it uniquely suited for embedding hierarchical and semantic relationships with low distortion.

| Property | Euclidean $\mathbb{R}^n$ | Hyperbolic $\mathbb{H}^n$ |
|----------|:-----------------------:|:------------------------:|
| Curvature | 0 | $< 0$ |
| Volume growth | $\Theta(r^n)$ | $\Theta(e^{(n-1)r})$ |
| Hierarchical capacity | Polynomial | Exponential |
| Distance growth | Linear | Exponential |

---

## The Poincare Ball Model

The Poincare ball is the open unit ball in $\mathbb{R}^n$:

$$\mathbb{B}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$$

### Riemannian Metric

The Poincare ball is endowed with the conformal Riemannian metric:

$$g_x = \lambda_x^2 \cdot I_n$$

where the **conformal factor** is:

$$\lambda_x = \frac{2}{1 - \|x\|^2}$$

This means distances stretch as points approach the boundary of the ball. The "infinitely expanding" boundary gives hyperbolic space its capacity advantage.

### Geodesic Distance

The geodesic distance between two points $u, v \in \mathbb{B}^n$ is:

$$d(u, v) = \text{arcosh}\left(1 + \frac{2\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}\right)$$

**Properties:**

- $d(u, v) \geq 0$, with equality iff $u = v$
- Symmetric: $d(u, v) = d(v, u)$
- Triangle inequality holds
- $d(0, v) = \text{arcosh}(1 + \frac{2\|v\|^2}{1 - \|v\|^2})$
- As $\|v\| \to 1$, $d(0, v) \to \infty$

---

## Core Operations

### Mobius Addition

Mobius addition generalizes vector addition to the Poincare ball:

$$u \oplus v = \frac{(1 + 2\langle u, v\rangle + \|v\|^2)u + (1 - \|u\|^2)v}{1 + 2\langle u, v\rangle + \|u\|^2\|v\|^2}$$

This is **not commutative** ($u \oplus v \neq v \oplus u$ in general), reflecting the non-Euclidean geometry. It is used as the building block for gyrovector space operations.

### Exponential Map

The exponential map projects a tangent vector $v$ at base point $p$ onto the manifold:

**At the origin** ($p = 0$):

$$\exp_0(v) = \tanh(\|v\|) \cdot \frac{v}{\|v\|}$$

**General case** ($p \neq 0$, gyrovector formulation):

$$\exp_p(v) = p \oplus \left(\tanh\left(\frac{\lambda_p \|v\|}{2}\right) \cdot \frac{v}{\|v\|}\right)$$

where $\lambda_p = \frac{2}{1 - \|p\|^2}$ is the conformal factor at $p$.

The exponential map is used to project Euclidean embeddings (e.g., from sentence transformers) into the Poincare ball.

### Logarithmic Map (Inverse)

The logarithmic map maps a manifold point back to the tangent space:

$$\log_p(q) = \frac{2}{\lambda_p} \cdot \frac{\text{arctanh}(\|-p \oplus q\|)}{\|-p \oplus q\|} \cdot (-p \oplus q)$$

---

## Embedding Training

### Loss Function

UCEF uses Riemannian SGD to minimize a contrastive loss:

$$\mathcal{L} = \sum_{(i,j) \in \mathcal{S}^+} \log d(u_i, u_j) + \sum_{(i,k) \in \mathcal{S}^-} \max(0, \delta - d(u_i, u_k))$$

where:
- $\mathcal{S}^+$ = similar document pairs (positive)
- $\mathcal{S}^-$ = dissimilar document pairs (negative)
- $\delta$ = margin hyperparameter
- $d(\cdot, \cdot)$ = geodesic distance

### Riemannian Gradient Descent

The gradient update in hyperbolic space differs from Euclidean SGD. The Riemannian gradient accounts for the metric tensor:

$$u \leftarrow \exp_u\left(-\eta \cdot \frac{(1 - \|u\|^2)^2}{4} \cdot \frac{\partial \mathcal{L}}{\partial u}\right)$$

Key steps:

1. Compute Euclidean gradient $\frac{\partial \mathcal{L}}{\partial u}$
2. Apply Riemannian correction: multiply by $\frac{(1-\|u\|^2)^2}{4}$
3. Project onto tangent space (automatic with the correction)
4. Exponential map back to the manifold
5. **Clip** the result to ensure $\|u\| < 1$ (stay inside the ball)

### Burn-in Phase

UCEF uses a burn-in phase with reduced learning rate ($\eta_{\text{burn}} = 0.001$ vs $\eta = 0.01$) for the first 20 epochs. This stabilizes early training by preventing large updates that push points near the boundary.

---

## Retrieval Algorithm

### Indexing

```python
def index(documents):
    for doc in documents:
        if doc.hyperbolic_embedding:
            point = doc.hyperbolic_embedding
        elif doc.euclidean_embedding:
            point = exponential_map(doc.euclidean_embedding)
        else:
            point = HyperbolicPoint.random(dim, max_norm=0.9)
        
        embeddings[doc.id] = point
    
    embedding_matrix = stack(all_coordinates)
```

### Batch Geodesic Distance

For a query point $q$ and embedding matrix $M \in \mathbb{R}^{n \times d}$:

```python
def batch_geodesic_distance(query, matrix):
    diff = matrix - query.coordinates          # (n, d)
    diff_norm_sq = sum(diff ** 2, axis=1)      # (n,)
    matrix_norm_sq = sum(matrix ** 2, axis=1)  # (n,)
    query_norm_sq = sum(query ** 2)            # scalar
    
    denominator = (1 - query_norm_sq) * (1 - matrix_norm_sq)
    denominator = clip(denominator, 1e-5, inf)
    
    inner = 1 + 2 * diff_norm_sq / denominator
    inner = clip(inner, 1.0, inf)  # arcosh domain
    
    return arccosh(inner)  # (n,)
```

**Complexity**: $O(n \cdot d)$ where $n$ = number of documents, $d$ = embedding dimension.

### Retrieval Pipeline

1. Embed query as Poincare ball point (via exponential map)
2. Compute geodesic distances to all indexed documents (vectorized)
3. Sort by ascending distance
4. Return top-k results

---

## Implementation: HyperbolicPoint

The `HyperbolicPoint` dataclass represents a point in the Poincare ball:

```python
@dataclass(frozen=True)
class HyperbolicPoint:
    coordinates: NDArray[np.float64]  # Must satisfy ||x|| < 1
    
    @property
    def dimension(self) -> int: ...
    
    @property
    def norm(self) -> float: ...
    
    @property
    def conformal_factor(self) -> float:
        """lambda_x = 2 / (1 - ||x||^2)"""
        return 2.0 / (1.0 - self.norm ** 2)
    
    def is_valid(self) -> bool:
        """Check if point lies within the open unit ball."""
        return self.norm < 1.0
    
    @classmethod
    def origin(cls, dim: int) -> HyperbolicPoint: ...
    
    @classmethod
    def random(cls, dim: int, max_norm: float = 0.9) -> HyperbolicPoint: ...
```

---

## Boundary Considerations

Points near the boundary ($\|x\| \to 1$) are numerically unstable:

| Norm | Conformal Factor $\lambda_x$ | Stability |
|------|:---------------------------:|:---------:|
| 0.0 | 2.0 | Excellent |
| 0.5 | 2.67 | Good |
| 0.9 | 10.53 | Acceptable |
| 0.95 | 20.41 | Caution |
| 0.99 | 100.5 | Unstable |
| 1.0 | $\infty$ | Invalid |

UCEF enforces `max_norm = 0.9` by default, keeping conformal factors below ~10.5. This provides a safety margin against numerical instability while still leveraging the exponential capacity of the boundary region.

---

## References

1. Nickel, M. & Kiela, D. (2017). "Poincare Embeddings for Learning Hierarchical Representations." *NeurIPS 2017*.
2. Nickel, M. & Kiela, D. (2018). "Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry." *ICML 2018*.
3. Ungar, A.A. (2008). "Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity." *World Scientific*.
4. Chamberlain, B.P. et al. (2019). "GRAND: Graph Neural Diffusion." *ICML 2021* (for Riemannian optimization details).

---

*Previous: [Architecture Overview](overview.md) | Next: [Quantum Selection](quantum.md)*
