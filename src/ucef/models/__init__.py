"""Model adapters for various LLM providers.

Provides concrete implementations of the ModelClient Protocol:
- BaseModelAdapter: Abstract base with retry/timeout logic
- OpenAIAdapter: GPT-4o, GPT-4-turbo, etc.
- AnthropicAdapter: Claude 3.5 Sonnet, Claude 3 Opus, etc.
- ZhipuAdapter: GLM-4, ChatGLM, etc.
- LocalAdapter: llama.cpp, vLLM, Ollama

Usage:
    from ucef.models import OpenAIAdapter
    adapter = OpenAIAdapter(model="gpt-4o", api_key="sk-...")
"""

from ucef.core.types import ModelClient
from ucef.models.base import BaseModelAdapter, AdapterConfig, GenerationStats

__all__ = [
    "ModelClient",
    "BaseModelAdapter",
    "AdapterConfig",
    "GenerationStats",
]

# Lazy imports — only fail at runtime if the provider SDK is not installed

def __getattr__(name: str):
    """Lazy-load provider adapters to avoid hard dependency on SDKs."""
    if name == "OpenAIAdapter":
        from ucef.models.openai import OpenAIAdapter
        return OpenAIAdapter
    elif name == "AnthropicAdapter":
        from ucef.models.anthropic import AnthropicAdapter
        return AnthropicAdapter
    elif name == "ZhipuAdapter":
        from ucef.models.zhipu import ZhipuAdapter
        return ZhipuAdapter
    elif name == "LocalAdapter":
        from ucef.models.local import LocalAdapter
        return LocalAdapter
    raise AttributeError(f"module 'ucef.models' has no attribute {name!r}")
