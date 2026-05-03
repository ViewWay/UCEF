# API Reference

## Core Classes

### UniversalContextSystem

The main entry point for UCEF.

```python
from ucef import UniversalContextSystem, UCEFConfig, Document

# Initialize
config = UCEFConfig()
system = UniversalContextSystem(
    model_client=adapter,
    model_name="glm-4-flash",
    config=config,
)
await system.initialize()

# Store documents
docs = [Document(id="doc1", text="..."), Document(id="doc2", text="...")]
await system.store_documents(docs)

# Query
result = await system.query("What is the main conclusion?")
print(result.overall_quality)  # 0.0 - 1.0
print(result.total_tokens)     # tokens used
print(result.context_blocks)   # list of ContextBlock
```

### Document

```python
from ucef import Document

doc = Document(
    id="unique_id",
    text="Document content...",
    metadata={"source": "wikipedia", "category": "science"},
)
```

### ContextBlock

A context block returned by the retrieval pipeline.

```python
# Access block properties
block = result.context_blocks[0]
block.text                  # Block content
block.relevance_score       # Relevance to query (0-1)
block.quantum_amplitude     # Quantum measurement amplitude
block.measurement_probability  # Collapse probability
block.token_count           # Approximate token count
```

### QueryResult

```python
result = await system.query("question")
result.overall_quality      # 0.3*R + 0.3*C + 0.2*H + 0.2*A
result.relevance            # Relevance score
result.completeness         # Completeness score
result.coherence            # Coherence score
result.accuracy             # Accuracy score
result.total_tokens         # Total tokens in context
result.retrieval_time_ms    # Retrieval latency
result.context_blocks       # List[ContextBlock]
```

## Configuration

### UCEFConfig

```python
from ucef import UCEFConfig

config = UCEFConfig()
config.hyperbolic.embedding_dim = 128
config.hyperbolic.curvature = -1.0
config.hyperbolic.max_norm = 0.9
config.hyperbolic.n_neighbors = 50
config.quantum.initial_amplitude = "equal"
config.quantum.entanglement_threshold = 0.3
config.quantum.measurement_method = "top_k"
config.quality.quality_threshold = 0.6
```

## Retrieval

### HyperbolicRetriever

```python
from ucef.retrieval.hyperbolic import HyperbolicRetriever

retriever = HyperbolicRetriever(config.hyperbolic)
await retriever.initialize()

# Add documents to index
await retriever.add_documents(documents)

# Retrieve nearest neighbors
results = await retriever.retrieve(query_embedding, top_k=10)
```

### QuantumSelector

```python
from ucef.retrieval.quantum import QuantumSelector

selector = QuantumSelector(config.quantum)
selected = selector.select(candidates, query_vector, token_budget=4000)
```

## Compression

### Adaptive Compressor

```python
from ucef.compression.adaptive import AdaptiveCompressor

compressor = AdaptiveCompressor()
compressed = compressor.compress(
    blocks=context_blocks,
    query="question",
    budget_tokens=4000,
    quality_retention=0.85,  # auto-selects strategy
)
```

## Quality

### QualityFeedbackLoop

```python
from ucef.quality.feedback import QualityFeedbackLoop

feedback = QualityFeedbackLoop(
    quality_threshold=0.6,
    max_iterations=3,
)
result = await feedback.refine(initial_result, system, query)
print(result.converged)  # True/False
print(result.iterations)  # number of refinement steps
```
