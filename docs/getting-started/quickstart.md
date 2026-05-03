---
title: Quickstart
---

# Quickstart Tutorial

This guide walks through UCEF's core workflow: initializing the system, storing documents, querying, and interpreting results. All code examples are runnable end-to-end.

---

## Prerequisites

```bash
pip install ucef
```

You'll also need a model client that implements the `ModelClient` protocol. UCEF provides a base adapter pattern you can implement for any LLM API.

---

## Step 1: Define a Model Client

UCEF is model-agnostic. You need to implement the `ModelClient` protocol:

```python
from ucef.core.types import ModelClient

class MyModelClient:
    """Simple model client for demonstration."""
    
    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base_url = base_url

    @property
    def model_name(self) -> str:
        return "my-model"

    @property
    def context_window(self) -> int:
        return 128_000  # Your model's native context window

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        # Call your LLM API here
        response = await call_your_api(prompt, max_tokens, temperature)
        return response

    async def count_tokens(self, text: str) -> int:
        # Use your tokenizer
        return len(text.split())  # Rough approximation
```

### Using with OpenAI-compatible APIs

```python
import openai

class OpenAIModelClient:
    """Model client for OpenAI-compatible APIs."""
    
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        windows = {
            "gpt-4o": 128_000,
            "gpt-4o-mini": 128_000,
            "gpt-3.5-turbo": 16_385,
        }
        return windows.get(self._model, 128_000)

    async def generate(self, prompt: str, max_tokens: int = 4096,
                       temperature: float = 0.7, **kwargs) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    async def count_tokens(self, text: str) -> int:
        # Use tiktoken for accurate counts
        import tiktoken
        enc = tiktoken.encoding_for_model(self._model)
        return len(enc.encode(text))
```

---

## Step 2: Initialize the System

```python
import asyncio
from ucef import UniversalContextSystem, UCEFConfig

async def setup():
    # Create model client
    client = OpenAIModelClient(model="gpt-4o", api_key="sk-...")

    # Create configuration (use defaults or customize)
    config = UCEFConfig(
        target_extended_context=1_000_000,  # Extend to 1M tokens
        hyperbolic=HyperbolicConfig(
            embedding_dim=128,
            n_neighbors=50,
        ),
        quantum=QuantumConfig(
            enabled=True,
            top_k_measurements=10,
        ),
        quality=QualityConfig(
            quality_threshold=0.75,
        ),
    )

    # Initialize system
    system = UniversalContextSystem(
        model_client=client,
        model_name="gpt-4o",
        config=config,
    )
    await system.initialize()

    return system
```

The `initialize()` call:

1. **Profiles the model** — Detects context window, measures performance curve, assesses quality retention
2. **Creates directories** — Sets up data, cache, warm, and cold storage paths
3. **Initializes compression engine** — Sets up the adaptive compressor with model-aware strategies
4. **Starts quality monitor** — Begins tracking query quality in a rolling window
5. **Prepares feedback loop** — Ready to refine low-quality results automatically

---

## Step 3: Store Documents

UCEF can store documents from various sources:

### Store individual documents

```python
from ucef import Document

# Store a single document
doc = await system.store_text(
    text="The Eiffel Tower is a wrought-iron lattice tower on the "
         "Champ de Mars in Paris, France. It is named after the "
         "engineer Gustave Eiffel, whose company designed and built "
         "the tower from 1887 to 1889.",
    doc_id="eiffel_001",
    metadata={
        "source": "wikipedia",
        "category": "landmarks",
        "language": "en",
    },
)
print(f"Stored document: {doc.id} ({doc.estimate_tokens()} tokens)")
```

### Store multiple documents at once

```python
documents = [
    Document(
        id="paper_001",
        text="Hyperbolic embeddings provide exponentially more capacity "
             "than Euclidean space for representing hierarchical data...",
        metadata={"source": "neurips_2017", "type": "research_paper"},
    ),
    Document(
        id="paper_002",
        text="The Minimum Description Length principle provides a "
             "rigorous framework for model selection based on "
             "information theory...",
        metadata={"source": "mit_press_2007", "type": "book"},
    ),
    Document(
        id="report_001",
        text="Quarterly financial report Q3 2025: Revenue increased "
             "by 15% year-over-year...",
        metadata={"source": "finance", "type": "report", "quarter": "Q3"},
    ),
]

stored_count = await system.store_documents(documents)
print(f"Stored {stored_count} documents")
```

### Store with pre-computed embeddings

```python
import numpy as np
from ucef import HyperbolicPoint

# If you have pre-computed embeddings
doc_with_embedding = Document(
    id="embedded_001",
    text="Document with known embedding...",
    euclidean_embedding=np.random.randn(128).astype(np.float64),
    hyperbolic_embedding=HyperbolicPoint.random(dim=128, max_norm=0.9),
)
await system.store_documents([doc_with_embedding])
```

---

## Step 4: Query for Context

### Basic query

```python
result = await system.query(
    "What is the Eiffel Tower and when was it built?"
)

print(f"Query: {result.query}")
print(f"Overall quality: {result.overall_quality:.3f}")
print(f"Context blocks selected: {len(result.context_blocks)}")
print(f"Total tokens: {result.total_tokens}")
print(f"Retrieval time: {result.retrieval_time_ms:.1f}ms")

# Print each context block
for i, block in enumerate(result.context_blocks, 1):
    print(f"\n[Block {i}] (score: {block.relevance_score:.3f})")
    print(f"  Tokens: {block.token_count}")
    print(f"  Measurement prob: {block.measurement_probability:.3f}")
    print(f"  Text: {block.text[:100]}...")
```

### Query with quality threshold override

```python
result = await system.query(
    "Explain the financial performance in Q3",
    top_k=20,                    # Retrieve more candidates
    quality_threshold=0.85,      # Higher quality bar
)
```

### Query with full model response

```python
# One-shot: query + generate response with extended context
response = await system.query_with_response(
    "Summarize the key findings from all documents"
)
print(response)
```

---

## Step 5: Interpret Results

### Quality Metrics

UCEF evaluates context quality across four dimensions:

```python
result = await system.query("What are the main topics?")

print(f"Relevance:    {result.relevance_score:.3f}    # How well blocks match query")
print(f"Completeness: {result.completeness_score:.3f}  # Coverage of query terms")
print(f"Coherence:    {result.coherence_score:.3f}    # Context consistency")
print(f"Accuracy:     {result.accuracy_score:.3f}     # Confidence in information")
print(f"Overall:      {result.overall_quality:.3f}    # Weighted combination")
```

The overall quality is computed as:

$$
Q = 0.30 \cdot R + 0.30 \cdot C_{complete} + 0.20 \cdot C_{coherent} + 0.20 \cdot A
$$

where $R$ = relevance, $C_{complete}$ = completeness, $C_{coherent}$ = coherence, $A$ = accuracy.

### QueryResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | The original query string |
| `context_blocks` | `List[ContextBlock]` | Selected context segments |
| `total_tokens` | `int` | Total tokens in selected context |
| `relevance_score` | `float` | Average block relevance (0-1) |
| `completeness_score` | `float` | Query term coverage (0-1) |
| `coherence_score` | `float` | Context consistency estimate (0-1) |
| `accuracy_score` | `float` | Information confidence (0-1) |
| `overall_quality` | `float` | Weighted quality score (0-1) |
| `retrieval_strategy` | `str` | Strategy used (adaptive/aggressive/etc.) |
| `compression_used` | `CompressionStrategy` | Compression level applied |
| `retrieval_time_ms` | `float` | Total pipeline latency |

### ContextBlock Fields

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | `str` | Source document identifier |
| `text` | `str` | Block text content |
| `relevance_score` | `float` | Relevance to query |
| `token_count` | `int` | Tokens in this block |
| `quantum_amplitude` | `complex` | Quantum state amplitude |
| `measurement_probability` | `float` | Born rule probability |

---

## Step 6: Monitor Quality

UCEF provides real-time quality monitoring:

```python
# Get quality statistics
stats = system.get_quality_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Mean quality: {stats['mean_quality']:.3f}")
print(f"P95 quality: {stats['p95_quality']:.3f}")
print(f"Quality degraded: {stats['quality_degraded']}")
```

### Query with feedback loop details

```python
# Get detailed feedback loop results
feedback = await system.query_with_feedback(
    "Explain the key concepts",
    quality_threshold=0.75,
)

print(f"Converged: {feedback.converged}")
print(f"Iterations: {feedback.iterations}")
print(f"Total improvement: {feedback.total_improvement:.3f}")

for step in feedback.steps:
    print(f"  Iteration {step.iteration}: "
          f"{step.quality_before:.3f} -> {step.quality_after:.3f} "
          f"({step.action.name}, {step.elapsed_ms:.1f}ms)")
```

---

## Step 7: System Statistics

```python
stats = await system.get_stats()
print(f"Model: {stats['model']}")
print(f"Context category: {stats['model_category']}")
print(f"Native window: {stats['native_context_window']} tokens")
print(f"Target context: {stats['target_context']} tokens")
print(f"Documents stored: {stats['documents_stored']}")
print(f"Quantum selection: {stats['quantum_selection_enabled']}")
```

---

## Complete Example

Here's a full runnable example that ties everything together:

```python
import asyncio
import numpy as np
from ucef import (
    UniversalContextSystem,
    UCEFConfig,
    Document,
    HyperbolicPoint,
)

# --- Mock model client for demo ---
class DemoClient:
    @property
    def model_name(self): return "demo-model"

    @property
    def context_window(self): return 4096

    async def generate(self, prompt, max_tokens=4096, **kw):
        return "This is a generated response."

    async def count_tokens(self, text):
        return len(text.split()) // 4

async def main():
    # Initialize
    system = UniversalContextSystem(
        model_client=DemoClient(),
        model_name="demo-model",
    )
    await system.initialize()

    # Store documents
    docs = [
        Document(id="doc1", text="Machine learning is a subset of AI..."),
        Document(id="doc2", text="Deep learning uses neural networks..."),
        Document(id="doc3", text="Natural language processing enables..."),
    ]
    await system.store_documents(docs)

    # Query
    result = await system.query("What is deep learning?")
    
    print(f"Quality: {result.overall_quality:.3f}")
    print(f"Blocks: {len(result.context_blocks)}")
    print(f"Latency: {result.retrieval_time_ms:.1f}ms")
    
    # Format context for model
    context = result.format_context()
    print(f"\nFormatted context:\n{context}")

asyncio.run(main())
```

---

## What's Next?

- [Configuration Reference](configuration.md) — Customize every aspect of UCEF
- [Architecture Overview](../architecture/overview.md) — Understand the mathematical foundations
- [API Reference](../api.md) — Full class and method documentation

---

*Previous: [Installation](installation.md) | Next: [Configuration](configuration.md)*
