# Retrieval API

## HyperbolicRetriever

```python
from ucef.retrieval.hyperbolic import HyperbolicRetriever
from ucef.core.config import HyperbolicConfig

retriever = HyperbolicRetriever(HyperbolicConfig(embedding_dim=8))
n = retriever.index(documents)   # Returns count indexed

# Retrieve by point
results = retriever.retrieve(query_point, top_k=10)
# → List[Tuple[Document, float]]  (doc, geodesic_distance)

# Retrieve by text
results = retriever.retrieve_by_text("machine learning", top_k=10)

# Manage index
retriever.add_document(doc)
retriever.remove_document("doc_id")
retriever.n_indexed           # Number of indexed docs
retriever.embedding_dim       # Embedding dimension
retriever.get_embedding("id") # Get embedding point
```

## QuantumSelector

```python
from ucef.retrieval.quantum import QuantumSelector
from ucef.core.config import QuantumConfig

selector = QuantumSelector(QuantumConfig(
    enabled=True,
    measurement_method="argmax",  # "argmax" | "sampling" | "top_k"
    use_interference=True,
))

blocks = selector.select(
    scored_documents,    # List[Tuple[Document, float]]
    query_weights=None,  # Optional NDArray weighting
    budget=token_budget, # TokenBudget constraint
)
# → List[ContextBlock]
```

## AdaptiveContextExtender

```python
from ucef.retrieval.adaptive import AdaptiveContextExtender

extender = AdaptiveContextExtender()
blocks = extender.extend(
    candidates=scored_docs,
    budget=budget,
    strategy="quantum",
)
```

## Fusion

```python
from ucef.retrieval.fusion import ReciprocalRankFusion, WeightedScoreFusion

# RRF
fusion = ReciprocalRankFusion(k=60)
merged = fusion.merge([result_list_1, result_list_2, result_list_3])

# Weighted
fusion = WeightedScoreFusion(weights=[0.5, 0.3, 0.2])
merged = fusion.merge([results_1, results_2, results_3])
```
