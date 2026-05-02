"""Model adapters — Phase 3 placeholder.

Concrete adapters (OpenAI, Anthropic, Zhipu, Local) will be implemented
in Phase 3. For now, users can implement the ModelClient Protocol directly.
"""

from ucef.core.types import ModelClient

__all__ = [
    "ModelClient",
]
