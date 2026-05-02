"""
Basic Usage Examples for UCEF

Demonstrates core features of the Universal Context Extension Framework.

References:
    - Nickel & Kiela, "Poincaré Embeddings", NeurIPS 2017
    - van Rijsbergen, "The Geometry of Information Retrieval", CUP 2004
"""

import asyncio

import numpy as np

from ucef import (
    UniversalContextSystem,
    UCEFConfig,
    ModelCapabilityProfiler,
    AdaptiveContextExtender,
    Document,
    HyperbolicPoint,
    QuantumState,
)


# ============================================================================
# Example 1: Basic Usage with Documents
# ============================================================================

async def example_1_basic_usage():
    """Basic usage: store documents and query with extended context."""
    from ucef.core.types import ModelClient

    # Create a simple mock model client for demonstration
    class MockModelClient:
        @property
        def model_name(self) -> str:
            return "glm-5.1"

        @property
        def context_window(self) -> int:
            return 200_000

        async def generate(self, prompt: str, **kwargs) -> str:
            return f"[Mock response to query embedded in prompt]"

        async def count_tokens(self, text: str) -> int:
            return len(text) // 4

    # Initialize UCEF
    client = MockModelClient()
    system = UniversalContextSystem(client, "glm-5.1")
    await system.initialize()

    # Store documents
    docs = [
        Document(id="doc_1", text="UCEF uses hyperbolic geometry for efficient semantic retrieval.", metadata={"source": "readme"}),
        Document(id="doc_2", text="The Poincare ball model maps hyperbolic space into a unit ball.", metadata={"source": "docs"}),
        Document(id="doc_3", text="Quantum-inspired selection uses superposition of context states.", metadata={"source": "docs"}),
    ]
    await system.store_documents(docs)

    # Query with automatic context extension
    query = "How does UCEF handle semantic retrieval?"
    result = await system.query(query)

    print(f"Query: {query}")
    print(f"Context blocks selected: {len(result.context_blocks)}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Quality score: {result.overall_quality:.3f}")


# ============================================================================
# Example 2: Model Profiling
# ============================================================================

async def example_2_model_profiling():
    """Example of profiling different models (known specs, no client needed)."""
    profiler = ModelCapabilityProfiler()

    models_to_profile = [
        "llama-7b",      # 4K context  -> small category
        "llama-13b",     # 32K context -> medium category
        "llama-3.1-70b", # 128K context -> large category
        "gpt-4o",        # 128K context -> large category
        "glm-5.1",       # 200K context -> large category
    ]

    print("=" * 60)
    print("Model Capability Profiles")
    print("=" * 60)

    for model_name in models_to_profile:
        # Known models can be profiled without a live client
        profile = await profiler.profile_model(None, model_name)

        print(f"\n{model_name.upper()}")
        print(f"  Native Context: {profile.native_context_window:,} tokens")
        print(f"  Category: {profile.context_category}")
        print(f"  Quality Retention: {profile.quality_retention:.2f}")
        print(f"  Recommended Strategy: {profile.recommended_strategy}")
        print(f"  Max Extended Context: {profile.max_extended_context:,} tokens")
        print(f"  Compression Ratio: {profile.optimal_compression_ratio:.0%}")


# ============================================================================
# Example 3: Hyperbolic Geometry
# ============================================================================

async def example_3_hyperbolic_geometry():
    """Demonstrate hyperbolic space operations for semantic retrieval."""
    from ucef.core.types import poincare_distance, mobius_add, exponential_map

    print("=" * 60)
    print("Hyperbolic Geometry: Poincare Ball Operations")
    print("=" * 60)

    # Create points in Poincare ball
    origin = HyperbolicPoint.origin(3)
    p1 = HyperbolicPoint.random(3, max_norm=0.5, seed=42)
    p2 = HyperbolicPoint.random(3, max_norm=0.7, seed=43)

    print(f"\nOrigin: {origin.coordinates}")
    print(f"Point 1: {p1.coordinates} (norm={p1.norm:.4f})")
    print(f"Point 2: {p2.coordinates} (norm={p2.norm:.4f})")

    # Geodesic distance
    d = poincare_distance(p1, p2)
    print(f"\nGeodesic distance d(p1, p2) = {d:.4f}")

    # Mobius addition
    p_sum = mobius_add(p1, p2)
    print(f"Mobius addition p1 + p2 = {p_sum.coordinates}")

    # Exponential map from tangent space
    v = np.array([0.5, -0.3, 0.8])
    mapped = exponential_map(v)
    print(f"Exponential map of {v} -> {mapped.coordinates} (norm={mapped.norm:.4f})")


# ============================================================================
# Example 4: Quantum-Inspired Context Selection
# ============================================================================

async def example_4_quantum_selection():
    """Demonstrate quantum-inspired superposition and measurement."""
    from ucef.core.types import DensityMatrix

    print("=" * 60)
    print("Quantum-Inspired Context Selection")
    print("=" * 60)

    # Create equal superposition over 5 candidate contexts
    n = 5
    labels = [f"document_{i}" for i in range(n)]
    state = QuantumState.equal_superposition(n, labels)
    print(f"\nEqual superposition |psi> = (1/sqrt({n})) Sum |ctx_i>")
    print(f"Probabilities: {state.probabilities}")

    # Apply relevance-based weighting (simulating query interaction)
    relevance_scores = np.array([0.9, 0.7, 0.3, 0.1, 0.05])
    weighted_state = QuantumState.from_probabilities(relevance_scores, labels)
    print(f"\nAfter query measurement (relevance-weighted):")
    print(f"Amplitudes: {weighted_state.amplitudes}")
    print(f"Probabilities: {weighted_state.probabilities}")

    # Density matrix for correlation analysis
    rho = DensityMatrix.from_pure_state(weighted_state)
    print(f"\nDensity matrix rho ({rho.dimension}x{rho.dimension}):")
    print(f"Valid (Tr=1, PSD): {rho.is_valid}")

    # Quantum similarity with a query vector
    query_vec = np.array([0.8, 0.6, 0.2, 0.05, 0.01], dtype=np.complex128)
    similarities = rho.quantum_similarity(query_vec)
    print(f"\nQuantum similarity with query: {similarities}")


# ============================================================================
# Example 5: Full Pipeline (Mock Model)
# ============================================================================

async def example_5_full_pipeline():
    """Complete pipeline using a mock model client."""

    class MockModelClient:
        @property
        def model_name(self) -> str:
            return "llama-7b"

        @property
        def context_window(self) -> int:
            return 4096

        async def generate(self, prompt: str, **kwargs) -> str:
            return "[Generated response based on extended context]"

        async def count_tokens(self, text: str) -> int:
            return len(text) // 4

    # 1. Create system
    client = MockModelClient()
    system = UniversalContextSystem(client, "llama-7b")
    await system.initialize()

    # 2. Store a larger corpus
    docs = [
        Document(id=f"doc_{i}", text=f"Research finding #{i}: " + " ".join(
            [f"token_{j}" for j in range(50)]
        ), metadata={"index": i})
        for i in range(20)
    ]
    stored = await system.store_documents(docs)
    print(f"Stored {stored} documents")

    # 3. Query with context extension
    stats = await system.get_stats()
    print(f"\nSystem stats: {stats}")

    result = await system.query("token_1 token_5")
    print(f"\nQuery result: {len(result.context_blocks)} blocks, {result.total_tokens} tokens")
    print(f"Quality: {result.overall_quality:.3f}")
    print(f"Strategy: {result.retrieval_strategy}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all examples."""
    print("UCEF Basic Usage Examples")
    print("=" * 60)

    await example_1_basic_usage()
    print("\n")
    await example_2_model_profiling()
    print("\n")
    await example_3_hyperbolic_geometry()
    print("\n")
    await example_4_quantum_selection()
    print("\n")
    await example_5_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
