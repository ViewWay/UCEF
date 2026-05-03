---
title: Quantum-Inspired Selection
---

# Quantum-Inspired Context Selection

UCEF's quantum selection module treats context selection as a quantum measurement problem. Documents are placed in superposition, inter-document correlations are captured through entanglement, and the query acts as a measurement operator that collapses the state to the most relevant subset.

---

## Theoretical Foundation

The quantum probability framework for information retrieval was introduced by van Rijsbergen (2004), who showed that classical retrieval models (Boolean, vector space, probabilistic) are special cases of a more general quantum formalism.

Key advantages of the quantum approach:

- **Superposition**: All candidates are considered simultaneously, not independently
- **Entanglement**: Inter-document correlations are captured via off-diagonal density matrix elements
- **Interference**: Constructive and destructive filtering naturally suppresses redundancy and boosts coherent clusters

---

## Quantum State Representation

### Superposition State

A collection of $n$ candidate documents is represented as a quantum state in superposition:

$$|\psi\rangle = \sum_{i=1}^{n} \alpha_i \, |\text{doc}_i\rangle$$

where $\alpha_i \in \mathbb{C}$ are complex amplitudes satisfying the normalization condition:

$$\sum_{i=1}^{n} |\alpha_i|^2 = 1$$

### Born Rule

The probability of "measuring" (selecting) document $i$ is given by the Born rule:

$$P(i) = |\alpha_i|^2$$

This connects the quantum amplitude to a classical probability, enabling probabilistic selection.

### Amplitude Initialization

UCEF supports three amplitude initialization strategies:

#### 1. Equal Superposition

$$\alpha_i = \frac{1}{\sqrt{n}} \quad \forall i$$

All documents are equally likely. Used when no prior information about relevance is available (exploratory queries).

#### 2. Relevance-Weighted

$$\alpha_i = \sqrt{p_i}, \quad p_i = \frac{\text{score}_i}{\sum_j \text{score}_j}$$

Amplitudes are proportional to the square root of retrieval scores. Higher-scored documents have higher measurement probability.

#### 3. Entropy-Weighted

$$\alpha_i = \sqrt{p_i}, \quad p_i \propto H(\text{doc}_i) \cdot \text{score}_i$$

where $H(\text{doc}_i) = -\sum_w p(w) \log_2 p(w)$ is the word-level information entropy of the document. This favors information-rich, diverse documents.

---

## Density Matrix

### Pure State

For a pure quantum state, the density matrix is:

$$\rho = |\psi\rangle\langle\psi| = \begin{pmatrix} \alpha_1 \\ \vdots \\ \alpha_n \end{pmatrix} \begin{pmatrix} \bar{\alpha}_1 & \cdots & \bar{\alpha}_n \end{pmatrix}$$

This is an $n \times n$ Hermitian positive semi-definite matrix with $\text{Tr}(\rho) = 1$.

### Mixed State with Entanglement

UCEF augments the pure-state density matrix with entanglement corrections based on document-document similarity:

$$\rho_{ij} \mathrel{+}= \begin{cases} \epsilon \cdot \text{sim}(\text{doc}_i, \text{doc}_j) & \text{if } \text{sim}(\text{doc}_i, \text{doc}_j) > \tau \\ 0 & \text{otherwise} \end{cases}$$

where:
- $\epsilon = 0.1$ is the entanglement strength
- $\text{sim}(\cdot, \cdot)$ is Jaccard similarity: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$
- $\tau$ is the `entanglement_threshold` (default 0.3)

**Physical interpretation**: When two documents share significant vocabulary, they are "entangled" — the selection of one influences the probability of selecting the other. This is encoded in the off-diagonal elements $\rho_{ij}$.

### Validity Checks

A valid density matrix must satisfy:

1. **Hermitian**: $\rho = \rho^\dagger$ (real eigenvalues)
2. **Positive semi-definite**: All eigenvalues $\geq 0$
3. **Unit trace**: $\text{Tr}(\rho) = 1$

UCEF's `DensityMatrix.is_valid` property checks all three conditions with numerical tolerance.

---

## Quantum Similarity Measurement

### Query as Measurement Operator

The user's query is encoded as a vector $|q\rangle$ in the same Hilbert space. The quantum similarity between the query and each candidate is computed as:

$$S(q, c_i) = \langle q | \rho | c_i \rangle$$

where $|c_i\rangle$ is the $i$-th basis state (one-hot vector for document $i$).

### Matrix Form

For all candidates simultaneously:

$$\vec{S} = \text{diag}(\rho \cdot |q\rangle)^*$$

When the query vector dimension doesn't match the density matrix dimension, UCEF falls back to the classical diagonal:

$$S_i = \rho_{ii} = |\alpha_i|^2$$

---

## Interference Filtering

### Quantum Interference

In quantum mechanics, when multiple paths are available, the total probability is **not** simply the sum of individual probabilities. Instead, amplitudes interfere:

$$P_{\text{total}} = |\alpha_1 + \alpha_2|^2 \neq |\alpha_1|^2 + |\alpha_2|^2$$

UCEF applies interference to modulate context selection probabilities.

### Interference Pattern

The interference between documents $i$ and $j$ is modeled as:

$$I(i) = \left|\sum_j \alpha_i \bar{\alpha}_j \cos(\theta_{ij})\right|^2$$

where $\theta_{ij}$ is the **phase difference** between documents $i$ and $j$, computed from their similarity:

$$\theta_{ij} = (1 - \text{sim}(i,j)) \cdot \frac{\pi}{2}$$

- **Similar documents** ($\text{sim} \to 1$): $\theta \to 0$, $\cos(\theta) \to 1$ → **constructive interference** (mutual boost)
- **Dissimilar documents** ($\text{sim} \to 0$): $\theta \to \pi/2$, $\cos(\theta) \to 0$ → **destructive interference** (mutual suppression)

### Modulated Probabilities

The final selection probability after interference is:

$$P_{\text{final}}(i) = \frac{P_{\text{measure}}(i) \cdot |I(i)|}{\sum_j P_{\text{measure}}(j) \cdot |I(j)|}$$

renormalized to ensure $\sum P_{\text{final}}(i) = 1$.

---

## State Collapse (Measurement)

After computing measurement probabilities, the quantum state "collapses" to the selected documents. UCEF supports three collapse methods:

### 1. Top-k Collapse (Default)

Select the $k$ documents with highest measurement probability:

$$\text{selected} = \text{argsort}(P_{\text{final}})[-k:]$$

Deterministic and reproducible. $k$ = `top_k_measurements` (default 10).

### 2. Argmax Collapse

Select all documents sorted by probability. The budget constraint is the only limiting factor.

### 3. Sampling Collapse

Probabilistic sampling without replacement according to the Born rule distribution:

$$P(\text{select } i) = P_{\text{final}}(i)$$

Each "measurement" simulates the inherent randomness of quantum observation. Good for promoting diversity.

### Budget Constraint

All collapse methods respect the token budget:

```python
for idx in selected_indices:
    if total_tokens + doc_tokens > budget.available_for_retrieval:
        continue  # Skip if over budget
    blocks.append(create_block(doc, idx))
    total_tokens += doc_tokens
```

---

## Implementation: QuantumState and DensityMatrix

### QuantumState

```python
@dataclass
class QuantumState:
    amplitudes: NDArray[np.complex128]
    labels: List[str]
    
    @property
    def probabilities(self) -> NDArray[np.float64]:
        """P(i) = |alpha_i|^2 (Born rule)"""
        return np.abs(self.amplitudes) ** 2
    
    @property
    def is_normalized(self) -> bool:
        """Check if sum |alpha_i|^2 = 1"""
        return abs(np.sum(np.abs(self.amplitudes)**2) - 1.0) < 1e-6
    
    def normalize(self) -> QuantumState: ...
    
    @classmethod
    def equal_superposition(cls, n, labels) -> QuantumState: ...
    
    @classmethod
    def from_probabilities(cls, probs, labels) -> QuantumState: ...
```

### DensityMatrix

```python
@dataclass
class DensityMatrix:
    matrix: NDArray[np.complex128]  # n x n Hermitian PSD
    
    @property
    def trace(self) -> complex: ...
    
    @property
    def is_valid(self) -> bool: ...  # Tr(rho)=1, PSD
    
    @classmethod
    def from_pure_state(cls, state) -> DensityMatrix:
        """rho = |psi><psi|"""
    
    @classmethod
    def from_mixed_states(cls, states) -> DensityMatrix:
        """rho = sum p_i |psi_i><psi_i|"""
    
    def quantum_similarity(self, query_vec) -> NDArray[np.float64]:
        """S(q, c) = <q|rho|c> for each basis state c"""
```

---

## Complexity Analysis

| Operation | Complexity | Notes |
|-----------|:----------:|-------|
| Build superposition | $O(n)$ | Normalize scores to probabilities |
| Build density matrix | $O(n^2)$ | Outer product + pairwise similarity |
| Quantum similarity | $O(n^2)$ | Matrix-vector product |
| Interference filtering | $O(n^2)$ | Pairwise similarity + outer product |
| Top-k collapse | $O(n \log n)$ | Argsort |
| **Total** | $O(n^2)$ | Dominated by density matrix operations |

For $n \leq 100$ candidates (typical after retrieval filtering), the quantum selection adds negligible overhead (~9ms mean, ~9.5ms p95 in benchmarks).

---

## When to Disable Quantum Selection

Set `QuantumConfig(enabled=False)` to fall back to classical top-k selection when:

- **Very few candidates** ($n < 5$): Quantum overhead not justified
- **Strict latency requirements** (< 5ms): Classical selection is ~1ms
- **Deterministic output required**: Sampling collapse introduces randomness

---

## References

1. van Rijsbergen, C.J. (2004). "The Geometry of Information Retrieval." *Cambridge University Press*.
2. Zuccon, G. et al. (2009). "The Quantum Probability Ranking Principle." *ICTIR 2009*.
3. Piwowarski, B. et al. (2010). "A Quantum Model of Information Retrieval." *ICTIR 2010*.
4. Sordoni, A. et al. (2013). "Modeling Information Retrieval with Quantum Entanglement." *ICTIR 2013*.

---

*Previous: [Hyperbolic Geometry](hyperbolic.md) | Next: [Compression](compression.md)*
