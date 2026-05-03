# Quick Start

## Basic Usage

```python
import asyncio
from ucef import UniversalContextSystem, UCEFConfig, Document

async def main():
    # 1. Create system with your model
    system = UniversalContextSystem(
        model_client=your_model_client,
        model_name="gpt-4o",
    )
    await system.initialize()

    # 2. Store documents
    docs = [
        Document(id="paper1", text="Long research paper text..."),
        Document(id="paper2", text="Another research paper..."),
    ]
    await system.store_documents(docs)

    # 3. Query
    result = await system.query("What are the key findings?")
    print(f"Quality: {result.overall_quality:.2%}")
    print(f"Tokens: {result.total_tokens}")
    print(f"Blocks: {len(result.context_blocks)}")
    print(result.format_context())

asyncio.run(main())
```

## Store Text Directly

```python
# Convenience method for single documents
doc = await system.store_text(
    "A very long document...",
    doc_id="my-doc",
    metadata={"source": "web", "date": "2026-05-03"},
)
```

## Query with Full Response

```python
# Get model response with extended context
response = await system.query_with_response("Summarize the key points")
print(response)
```

## Quality Feedback Loop

```python
from ucef.quality.feedback import QualityFeedbackLoop

# Explicit feedback loop with details
feedback = await system.query_with_feedback(
    "What are the main arguments?",
)
print(f"Converged: {feedback.converged}")
print(f"Iterations: {feedback.iterations}")
print(f"Total improvement: {feedback.total_improvement:.2%}")
```

## Monitor Quality

```python
# Get quality statistics
stats = system.get_quality_stats()
print(stats)

# Get system stats
stats = await system.get_stats()
print(f"Model: {stats['model']}")
print(f"Documents stored: {stats['documents_stored']}")
print(f"Context window: {stats['native_context_window']}")
```

## Run Experiments

```bash
# Simulated experiments (no API key needed)
python experiments/simulated_experiment.py

# Real experiments with API keys
export OPENAI_API_KEY=sk-...
python experiments/real_experiment.py -b all -m gpt-4o -n 100
```
