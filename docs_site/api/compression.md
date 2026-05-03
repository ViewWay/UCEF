# Compression API

## AdaptiveCompressor

```python
from ucef.compression.adaptive import AdaptiveCompressor
from ucef.core.types import CompressionStrategy

compressor = AdaptiveCompressor(config)

# Auto-select strategy
compressed, result = await compressor.compress(
    blocks, budget, query="query text",
)

# Explicit strategy
compressed, result = await compressor.compress(
    blocks, budget,
    strategy=CompressionStrategy.AGGRESSIVE,
    quality_retention=0.8,
)

result.original_tokens   # Token count before compression
result.compressed_tokens # Token count after compression
```

## MDLCompressor

```python
from ucef.compression.mdl import MDLCompressor

compressor = MDLCompressor(description_length_weight=0.5)
compressed, result = compressor.compress_blocks(blocks, budget=1000, query="query")

# Single text compression
text = compressor.compress_block_text(
    "Long text...", target_ratio=0.3, query="search terms",
)
```

## EntropyCompressor

```python
from ucef.compression.entropy import EntropyCompressor

compressor = EntropyCompressor(
    redundancy_threshold=0.7,
    diversity_weight=0.3,
)
compressed, result = compressor.compress_blocks(blocks, budget=500, query="query")

# Utility methods
entropy = compressor.block_entropy(block)
redundancy = compressor.redundancy_score(block_a, block_b)
```

## TaskAwareCompressor

```python
from ucef.compression.task_aware import TaskAwareCompressor

compressor = TaskAwareCompressor(model_client=client)
compressed, result = compressor.compress_blocks(blocks, budget=2000, query="query")

# Extract key sentences
sentences = compressor.extract_key_sentences(text, query, top_k=5)
```

## Compression Strategies

| Strategy | Enum | Retention | Use Case |
|----------|------|-----------|----------|
| Aggressive | `CompressionStrategy.AGGRESSIVE` | ~10% | Small context (4K–32K) |
| Moderate | `CompressionStrategy.MODERATE` | ~30% | Medium context (32K–128K) |
| Light | `CompressionStrategy.LIGHT` | ~50% | Large context (128K+) |
| Adaptive | `CompressionStrategy.ADAPTIVE` | Auto | Auto-selected |
