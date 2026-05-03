# Core API

## UniversalContextSystem

Main entry point for UCEF.

```python
from ucef import UniversalContextSystem, UCEFConfig, Document

system = UniversalContextSystem(
    model_client=client,      # ModelClient protocol
    model_name="gpt-4o",      # Model identifier
    config=UCEFConfig(),       # Optional configuration
)
await system.initialize()
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `None` | Initialize all subsystems |
| `store_documents(docs)` | `int` | Store documents, returns count |
| `store_text(text, doc_id, metadata)` | `Document` | Convenience for single text |
| `query(query, top_k, quality_threshold)` | `QueryResult` | Execute context extension query |
| `query_with_response(query)` | `str` | Query + generate model response |
| `query_with_feedback(query)` | `FeedbackResult` | Query with explicit feedback details |
| `get_stats()` | `dict` | System statistics |
| `get_quality_stats()` | `dict` | Quality monitor statistics |

## Document

```python
from ucef import Document

doc = Document(
    id="paper1",
    text="Long document text...",
    metadata={"source": "arxiv"},
)
doc.estimate_tokens()  # Returns token count
```

## ContextBlock

```python
from ucef import ContextBlock

block = ContextBlock(
    document_id="paper1",
    text="Selected context text",
    relevance_score=0.95,
    token_count=150,
)
```

## QueryResult

```python
result = await system.query("What is UCEF?")

result.query                # Original query
result.context_blocks       # List[ContextBlock]
result.total_tokens         # Total tokens in context
result.overall_quality      # 0.0–1.0 quality score
result.relevance_score      # Relevance dimension
result.completeness_score   # Completeness dimension
result.coherence_score      # Coherence dimension
result.accuracy_score       # Accuracy dimension
result.retrieval_strategy   # Strategy used
result.retrieval_time_ms    # Retrieval latency

result.format_context()     # Formatted context string
```

## TokenBudget

```python
from ucef import TokenBudget

budget = TokenBudget.from_context_window(
    context_window=128000,
    system_tokens=500,
    conversation_tokens=1000,
    response_tokens=2000,
)

budget.available_for_retrieval  # Tokens available for context
```
