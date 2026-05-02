# UCEF Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/extend-Context-System.git
cd extend-Context-System

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Basic Usage

### 1. Initialize System for Any Model

```python
from ucef import UniversalContextSystem

# Works with OpenAI
from openai import OpenAI
client = OpenAI(api_key="your-key")
system = UniversalContextSystem(client, "gpt-4o")

# Works with Anthropic
from anthropic import Anthropic
client = Anthropic(api_key="your-key")
system = UniversalContextSystem(client, "claude-3-5-sonnet")

# Works with local models
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("llama-7b")
system = UniversalContextSystem(model, "llama-7b")
```

### 2. Store Documents

```python
# Load your documents
documents = [
    {"id": "doc1", "text": "...", "metadata": {...}},
    {"id": "doc2", "text": "...", "metadata": {...}},
    # ... up to 1M+ tokens!
]

# Store in system
await system.store_documents(documents)
```

### 3. Query with Quality Preservation

```python
# Query automatically uses optimal strategy
response = await system.query("Your complex query")

# Quality is automatically preserved
# Output comparable to top models!
print(response)
```

## Adaptive Strategies

UCEF automatically selects optimal strategy:

### Small Context (4K-32K)
```python
# Llama-7B (4K) → 1M+ tokens
# Strategy: Aggressive compression + Precision retrieval
system = UniversalContextSystem(llama_7b, "llama-7b")
```

### Medium Context (32K-128K)
```python
# Llama-13B (32K) → 1M+ tokens
# Strategy: Moderate compression + Hyperbolic retrieval
system = UniversalContextSystem(llama_13b, "llama-13b")
```

### Large Context (128K-200K)
```python
# GLM-5.1 (200K) → 1M+ tokens
# Strategy: Light compression + Structure preservation
system = UniversalContextSystem(glm_5_1, "glm-5.1")
```

## Running Experiments

```bash
# Run all benchmarks
bash experiments/run_all.sh

# Run specific benchmark
python scripts/eval/run_benchmark.py --benchmark needle

# Quality comparison
python scripts/eval/quality_comparison.py --models llama-7b,gpt-4o
```

## Next Steps

1. Read [API Reference](api/reference.md)
2. Check [Tutorials](tutorials/)
3. Review [Theoretical Background](theory/)
4. Run [Experiments](../experiments/)

## Support

- GitHub Issues
- Documentation
- Email: ucef@example.com
