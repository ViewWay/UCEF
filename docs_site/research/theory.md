# Theoretical Background

## Hyperbolic Geometry

UCEF uses the Poincaré ball model of hyperbolic space for document embedding and retrieval.

### Why Hyperbolic Space?

Hierarchical data (taxonomies, ontologies, document trees) is naturally suited to hyperbolic geometry:

- **Exponential capacity**: For dimension $d$, hyperbolic space has exponentially more room than Euclidean space
- **Hierarchy encoding**: Root-to-leaf paths map to origin-to-boundary distances
- **Compact representation**: Tree structures with $n$ nodes require only $O(\log n)$ distortion

### Key Equations

Geodesic distance:

$$d(u, v) = \text{arcosh}\left(1 + \frac{2\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right)$$

Exponential map at origin:

$$\exp_0(v) = \tanh(\|v\|) \cdot \frac{v}{\|v\|}$$

Reference: Nickel & Kiela, NeurIPS 2017.

## Quantum Probability

UCEF applies quantum probability theory to context selection.

### Key Concepts

- **Superposition**: Documents exist in superposition until measured
- **Density Matrix**: $\rho = |\psi\rangle\langle\psi|$ encodes document correlations
- **Measurement**: Query acts as measurement operator, collapsing superposition
- **Interference**: Constructive/destructive interference filters redundant context

### Born Rule

$$P(i) = |\alpha_i|^2$$

Measurement probability equals squared amplitude magnitude.

Reference: van Rijsbergen, "The Geometry of Information Retrieval", CUP 2004.

## Information Theory

### Minimum Description Length (MDL)

$$\text{MDL} = L(\text{context}) + L(\text{query} \mid \text{context})$$

Select context that minimizes total description cost.

Reference: Grünwald, "The MDL Principle", MIT Press 2007.

### Maximum Entropy

$$H = -\sum_i p_i \log_2 p_i$$

Maximize information entropy to ensure diverse, non-redundant context selection.

Reference: Jaynes, 1957.

## Statistical Mechanics

### Free Energy

$$F = E - T \cdot S$$

Map context selection to free energy minimization: energy = relevance cost, entropy = diversity, temperature = exploration parameter.

### Renormalization Group

Multi-scale coarse-graining inspired by Wilson's RG:

$$\text{Full text} \xrightarrow{\text{RG}} \text{Sentences} \xrightarrow{\text{RG}} \text{Summaries}$$

Progressively removes irrelevant detail at each scale.
