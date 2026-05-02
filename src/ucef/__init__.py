"""
Universal Context Extension Framework (UCEF)

Breaking the Context Barrier: Model-Agnostic Infinite Context with Quality Preservation

References:
    - Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations", NeurIPS 2017
    - van Rijsbergen, "The Geometry of Information Retrieval", Cambridge University Press, 2004
    - Grünwald, "The Minimum Description Length Principle", MIT Press, 2007
"""

__version__ = "0.1.0"
__author__ = "UCEF Team"

from ucef.core.config import UCEFConfig
from ucef.core.system import UniversalContextSystem
from ucef.core.types import (
    CompressionStrategy,
    ContextBlock,
    ContextCategory,
    Document,
    HyperbolicPoint,
    ModelProfile,
    QuantumState,
    QueryResult,
    TokenBudget,
)
from ucef.quality.profiler import ModelCapabilityProfiler
from ucef.retrieval.adaptive import AdaptiveContextExtender

__all__ = [
    "UniversalContextSystem",
    "UCEFConfig",
    "ModelCapabilityProfiler",
    "AdaptiveContextExtender",
    "CompressionStrategy",
    "ContextBlock",
    "ContextCategory",
    "Document",
    "HyperbolicPoint",
    "ModelProfile",
    "QuantumState",
    "QueryResult",
    "TokenBudget",
]
