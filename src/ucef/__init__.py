"""
Universal Context Extension Framework (UCEF)

Breaking the Context Barrier: Model-Agnostic Infinite Context with Quality Preservation

References:
    - Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations", NeurIPS 2017
    - van Rijsbergen, "The Geometry of Information Retrieval", Cambridge University Press, 2004
    - Grünwald, "The Minimum Description Length Principle", MIT Press, 2007
"""

__version__ = "0.3.0"
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
from ucef.quality.monitor import QualityMonitor
from ucef.quality.feedback import QualityFeedbackLoop, FeedbackResult
from ucef.retrieval.adaptive import AdaptiveContextExtender
from ucef.compression.adaptive import AdaptiveCompressor
from ucef.physics.thermodynamic import ThermodynamicModel
from ucef.physics.quantum_field import RenormalizationGroup
from ucef.models.base import BaseModelAdapter, AdapterConfig

__all__ = [
    "UniversalContextSystem",
    "UCEFConfig",
    "ModelCapabilityProfiler",
    "QualityMonitor",
    "QualityFeedbackLoop",
    "FeedbackResult",
    "AdaptiveContextExtender",
    "AdaptiveCompressor",
    "ThermodynamicModel",
    "RenormalizationGroup",
    "BaseModelAdapter",
    "AdapterConfig",
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
