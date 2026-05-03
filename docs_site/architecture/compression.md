# Adaptive Compression

Three compression strategies that adapt to model capabilities and context budget.

## Strategies

### Aggressive — MDL Principle

$$\text{MDL} = L(\text{context}) + L(\text{query} \mid \text{context})$$

Minimize description cost subject to token budget. Retains **~10%** of original.

Best for small context models (4K–32K) where every token counts.

### Moderate — Maximum Entropy

$$H(\text{selected}) = -\sum_i p_i \log_2 p_i$$

Maximize information entropy while respecting budget. Retains **~30%**. Balances information density with diversity using MMR (Maximal Marginal Relevance).

Best for medium context models (32K–128K).

### Light — Task-Aware Extraction

Query-directed sentence extraction inspired by ATACompressor. Retains **~50%** with **~92%** quality preservation.

Best for large context models (128K+).

## Strategy Selection

Automatic strategy selection based on model quality retention:

| Quality Retention | Strategy |
|-------------------|----------|
| ≥ 0.9 | Light (50%) |
| 0.7 – 0.9 | Moderate (30%) |
| < 0.7 | Aggressive (10%) |

## API

```python
from ucef.compression.adaptive import AdaptiveCompressor
from ucef.core.types import CompressionStrategy, TokenBudget

compressor = AdaptiveCompressor(config)

# Auto-select strategy
compressed, result = await compressor.compress(
    blocks, budget,
    strategy=CompressionStrategy.ADAPTIVE,
    query="user query",
)

# Explicit strategy
compressed, result = await compressor.compress(
    blocks, budget,
    strategy=CompressionStrategy.AGGRESSIVE,
)
```

## Experiment Results

| Strategy | Retained | Quality |
|----------|----------|---------|
| Aggressive (MDL) | ~10% | ~92.6% |
| Moderate (Entropy) | ~28% | ~72.0% |
| Light (Task-Aware) | ~50% | ~69.0% |

## References

- Grünwald, "The Minimum Description Length Principle", MIT Press, 2007
- Jaynes, "Information Theory and Statistical Mechanics", 1957
- Jiang et al., "LLMLingua", EMNLP 2023
