# Model Adapters API

## BaseModelAdapter

All adapters inherit from `BaseModelAdapter` which provides retry logic, timeout handling, and token counting.

```python
from ucef.models.base import BaseModelAdapter, AdapterConfig

config = AdapterConfig(
    model_name="gpt-4o",
    context_window=128000,
    max_retries=3,
    timeout=60.0,
)
```

## OpenAI Adapter

```python
from ucef.models.openai import OpenAIAdapter

adapter = OpenAIAdapter(
    model="gpt-4o",
    api_key="sk-...",  # Or set OPENAI_API_KEY env var
)

response = await adapter.generate("Hello, world!")
tokens = await adapter.count_tokens("Some text")
```

## Anthropic Adapter

```python
from ucef.models.anthropic import AnthropicAdapter

adapter = AnthropicAdapter(
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",  # Or set ANTHROPIC_API_KEY env var
)

response = await adapter.generate("Hello, world!")
```

## Zhipu Adapter

```python
from ucef.models.zhipu import ZhipuAdapter

adapter = ZhipuAdapter(
    model="glm-4",
    api_key="...",  # Or set ZHIPU_API_KEY env var
)

response = await adapter.generate("你好世界")
```

## Local Adapter

```python
from ucef.models.local import LocalAdapter

adapter = LocalAdapter(config=AdapterConfig(
    model_name="llama-7b",
    context_window=4096,
))

response = await adapter.generate("Hello")
```

Supports llama.cpp, vLLM, and Ollama backends.

## Statistics

```python
stats = adapter.get_stats()
# → {"total_calls": N, "mean_latency_ms": ..., "min_latency_ms": ..., "max_latency_ms": ...}
```
