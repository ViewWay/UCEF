"""
UCEF Core Type System

Mathematical type definitions for the Universal Context Extension Framework.
Grounded in hyperbolic geometry, quantum probability theory, and information theory.

References:
    - Nickel & Kiela, "Poincaré Embeddings", NeurIPS 2017
    - van Rijsbergen, "The Geometry of Information Retrieval", CUP 2004
    - Grünwald, "The Minimum Description Length Principle", MIT Press 2007
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)
import numpy as np
from numpy.typing import NDArray


# ──────────────────────────────────────────────────────────────────────────────
# 1. Geometric Types — Hyperbolic Space (Poincaré Ball Model)
# ──────────────────────────────────────────────────────────────────────────────

class CurvatureType(Enum):
    """Space curvature classification."""
    EUCLIDEAN = "euclidean"      # Flat space, R^n
    HYPERBOLIC = "hyperbolic"    # Negative curvature, H^n (Poincaré ball)
    SPHERICAL = "spherical"      # Positive curvature, S^n


@dataclass(frozen=True)
class HyperbolicPoint:
    """
    A point in the Poincaré ball model of hyperbolic space.

    The Poincaré ball B^n = {x ∈ R^n : ||x|| < 1} is endowed with the
    Riemannian metric:
        g_x = λ_x² · I,  where λ_x = 2 / (1 - ||x||²)

    Reference: Nickel & Kiela (2017), Eq. 1
    """
    coordinates: NDArray[np.float64]  # Must satisfy ||x|| < 1

    @property
    def dimension(self) -> int:
        return len(self.coordinates)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.coordinates))

    @property
    def conformal_factor(self) -> float:
        """λ_x = 2 / (1 - ||x||²), the local scaling factor."""
        return 2.0 / (1.0 - self.norm ** 2)

    def is_valid(self) -> bool:
        """Check if point lies within the open unit ball."""
        return self.norm < 1.0

    @classmethod
    def origin(cls, dim: int) -> HyperbolicPoint:
        """Create the origin point (zero vector) in Poincaré ball."""
        return cls(coordinates=np.zeros(dim, dtype=np.float64))

    @classmethod
    def random(cls, dim: int, max_norm: float = 0.9, seed: Optional[int] = None) -> HyperbolicPoint:
        """Generate a random point in the Poincaré ball."""
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim)
        vec = vec / np.linalg.norm(vec) * max_norm * rng.random()
        return cls(coordinates=vec)


def poincare_distance(u: HyperbolicPoint, v: HyperbolicPoint) -> float:
    """
    Compute the geodesic distance in the Poincaré ball model.

    d(u, v) = arcosh(1 + 2 · ||u - v||² / ((1 - ||u||²)(1 - ||v||²)))

    Reference: Nickel & Kiela (2017), Eq. 3
    """
    u_norm_sq = u.norm ** 2
    v_norm_sq = v.norm ** 2
    diff_norm_sq = float(np.sum((u.coordinates - v.coordinates) ** 2))

    denominator = (1.0 - u_norm_sq) * (1.0 - v_norm_sq)
    if denominator <= 0:
        return float('inf')

    inner = 1.0 + 2.0 * diff_norm_sq / denominator
    return float(np.arccosh(max(inner, 1.0)))


def mobius_add(u: HyperbolicPoint, v: HyperbolicPoint) -> HyperbolicPoint:
    """
    Möbius addition in the Poincaré ball.

    u ⊕ v = ((1 + 2⟨u,v⟩ + ||v||²)u + (1 - ||u||²)v) / (1 + 2⟨u,v⟩ + ||u||²||v||²)

    Reference: Ungar (2008), Gyrovector spaces
    """
    u_sq = u.norm ** 2
    v_sq = v.norm ** 2
    uv = float(np.dot(u.coordinates, v.coordinates))

    denom = 1.0 + 2.0 * uv + u_sq * v_sq
    if abs(denom) < 1e-10:
        denom = 1e-10

    numerator = ((1.0 + 2.0 * uv + v_sq) * u.coordinates +
                 (1.0 - u_sq) * v.coordinates)
    return HyperbolicPoint(coordinates=numerator / denom)


def exponential_map(v: NDArray[np.float64], base: Optional[HyperbolicPoint] = None) -> HyperbolicPoint:
    """
    Exponential map from tangent space to Poincaré ball.

    Maps a tangent vector v at point base to a point on the manifold.
    If base is None, maps from the origin.

    At origin:
        exp_0(v) = tanh(||v||) · v / ||v||

    General case (gyrovector formulation):
        exp_base(v) = base ⊕ (tanh(λ_base · ||v|| / 2) · v / ||v||)

    Reference: Nickel & Kiela (2017), Supplementary Eq. S4; Ungar (2008)
    """
    if base is None:
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            return HyperbolicPoint.origin(len(v))
        return HyperbolicPoint(coordinates=np.tanh(norm_v) * v / norm_v)

    # General case: exp_base(v) = base ⊕ tanh(λ_base · ||v|| / 2) · (v / ||v||)
    lambda_base = base.conformal_factor
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        return base

    direction = v / v_norm
    second_term = np.tanh(lambda_base * v_norm / 2.0) * direction
    return mobius_add(base, HyperbolicPoint(coordinates=second_term))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Quantum-Inspired Types — Context State Representation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantumState:
    """
    Quantum-inspired representation of candidate context states.

    A context state is a superposition of all candidate contexts:
        |ψ⟩ = Σᵢ αᵢ |context_i⟩

    where αᵢ are complex amplitudes satisfying Σ|αᵢ|² = 1.

    Reference: van Rijsbergen, "The Geometry of Information Retrieval" (2004)
    """
    amplitudes: NDArray[np.complex128]  # Complex coefficients αᵢ
    labels: List[str] = field(default_factory=list)  # Context identifiers

    @property
    def n_states(self) -> int:
        return len(self.amplitudes)

    @property
    def probabilities(self) -> NDArray[np.float64]:
        """P(i) = |αᵢ|² — Born rule measurement probabilities."""
        return np.abs(self.amplitudes) ** 2

    @property
    def is_normalized(self) -> bool:
        """Check if Σ|αᵢ|² = 1 (normalization condition)."""
        return abs(np.sum(np.abs(self.amplitudes) ** 2) - 1.0) < 1e-6

    def normalize(self) -> QuantumState:
        """
        Normalize the state to satisfy Σ|αᵢ|² = 1.

        Raises:
            ValueError: If all amplitudes are zero (ill-defined state).
        """
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm < 1e-10:
            raise ValueError(
                "Cannot normalize a zero-amplitude quantum state. "
                "Provide non-zero amplitudes."
            )
        return QuantumState(
            amplitudes=self.amplitudes / norm,
            labels=self.labels
        )

    @classmethod
    def equal_superposition(cls, n: int, labels: Optional[List[str]] = None) -> QuantumState:
        """
        Create an equal superposition state: |ψ⟩ = (1/√n) Σ |context_i⟩

        All contexts are equally likely before measurement.
        """
        amp = np.ones(n, dtype=np.complex128) / np.sqrt(n)
        return cls(
            amplitudes=amp,
            labels=labels or [f"ctx_{i}" for i in range(n)]
        )

    @classmethod
    def from_probabilities(cls, probs: NDArray[np.float64],
                           labels: Optional[List[str]] = None) -> QuantumState:
        """Create state from probability distribution (real amplitudes)."""
        sqrt_p = np.sqrt(probs.astype(np.float64))
        return cls(
            amplitudes=sqrt_p.astype(np.complex128),
            labels=labels or [f"ctx_{i}" for i in range(len(probs))]
        )


@dataclass
class DensityMatrix:
    """
    Density matrix representation of a mixed context state.

    ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|

    Encodes both probability distributions and inter-context correlations
    (entanglement between document pairs).

    Reference: Zuccon et al., "Quantum Probability Ranking Principle" (2009)
    """
    matrix: NDArray[np.complex128]  # n × n Hermitian positive semi-definite

    @property
    def dimension(self) -> int:
        return self.matrix.shape[0]

    @property
    def trace(self) -> complex:
        return np.trace(self.matrix)

    @property
    def is_valid(self) -> bool:
        """Check if Tr(ρ) = 1 and ρ is positive semi-definite (allowing numerical tolerance)."""
        trace_ok = abs(self.trace - 1.0) < 1e-6
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        # PSD check: all eigenvalues >= 0 (with tolerance for floating point)
        psd_ok = np.all(eigenvalues >= -1e-10)
        return trace_ok and psd_ok

    @classmethod
    def from_pure_state(cls, state: QuantumState) -> DensityMatrix:
        """Construct ρ = |ψ⟩⟨ψ| from a pure quantum state."""
        vec = state.amplitudes.reshape(-1, 1)
        return cls(matrix=vec @ vec.conj().T)

    @classmethod
    def from_mixed_states(cls, states: List[Tuple[float, QuantumState]]) -> DensityMatrix:
        """
        Construct ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ| from a mixture of states.
        """
        dim = states[0][1].n_states
        rho = np.zeros((dim, dim), dtype=np.complex128)
        for prob, state in states:
            vec = state.amplitudes.reshape(-1, 1)
            rho += prob * (vec @ vec.conj().T)
        return cls(matrix=rho)

    def quantum_similarity(self, query_vec: NDArray[np.complex128]) -> NDArray[np.float64]:
        """
        Compute quantum similarity: S(q, c) = ⟨q|ρ|c⟩ for each basis state.

        This generalizes classical dot-product similarity by incorporating
        quantum correlations (off-diagonal density matrix elements).
        """
        dim = self.dimension
        similarities = np.zeros(dim, dtype=np.float64)
        for i in range(dim):
            e_i = np.zeros(dim, dtype=np.complex128)
            e_i[i] = 1.0
            similarities[i] = float(np.real(query_vec.conj() @ self.matrix @ e_i))
        return similarities


# ──────────────────────────────────────────────────────────────────────────────
# 3. Information-Theoretic Types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenBudget:
    """
    Token budget allocation for context selection.

    Models the constraint that total selected context must fit within
    the model's native context window.
    """
    total: int                           # Total available tokens
    system_prompt: int = 0               # Reserved for system prompt
    conversation: int = 0                # Reserved for conversation history
    retrieved_context: int = 0           # Reserved for retrieved context
    response_buffer: int = 0             # Reserved for model response

    @property
    def available_for_retrieval(self) -> int:
        """Tokens available for retrieved context after other reservations."""
        return max(0, self.total - self.system_prompt -
                   self.conversation - self.response_buffer)

    @classmethod
    def from_context_window(
        cls,
        context_window: int,
        system_tokens: int = 500,
        conversation_tokens: int = 1000,
        response_tokens: int = 2000,
    ) -> TokenBudget:
        """Create budget from model's context window size."""
        return cls(
            total=context_window,
            system_prompt=system_tokens,
            conversation=conversation_tokens,
            response_buffer=response_tokens,
            retrieved_context=context_window - system_tokens -
                               conversation_tokens - response_tokens,
        )


@dataclass
class CompressionResult:
    """Result of a context compression operation."""
    original_tokens: int
    compressed_tokens: int

    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed to original tokens."""
        return self.compressed_tokens / max(self.original_tokens, 1)

    @property
    def reduction_percentage(self) -> float:
        """Percentage of tokens removed."""
        return (1.0 - self.compression_ratio) * 100


# ──────────────────────────────────────────────────────────────────────────────
# 4. Document & Context Types
# ──────────────────────────────────────────────────────────────────────────────

class ContextCategory(Enum):
    """Classification of model context window sizes."""
    SMALL = "small"       # 4K - 32K tokens
    MEDIUM = "medium"     # 32K - 128K tokens
    LARGE = "large"       # 128K - 200K tokens
    XLARGE = "xlarge"     # 200K+ tokens


class CompressionStrategy(Enum):
    """Available compression strategies."""
    AGGRESSIVE = "aggressive"    # 10% retention (small context models)
    MODERATE = "moderate"        # 30% retention (medium context models)
    LIGHT = "light"              # 50% retention (large context models)
    ADAPTIVE = "adaptive"        # Dynamically determined


@dataclass
class Document:
    """
    A document in the UCEF system.

    Supports both Euclidean and hyperbolic embedding representations.
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Embeddings
    euclidean_embedding: Optional[NDArray[np.float64]] = None
    hyperbolic_embedding: Optional[HyperbolicPoint] = None

    # Token accounting
    token_count: int = 0

    # Quality metrics
    relevance_score: float = 0.0
    information_density: float = 0.0

    def estimate_tokens(self) -> int:
        """Estimate token count if not provided."""
        if self.token_count > 0:
            return self.token_count
        # Rough estimate: ~4 characters per token for English
        self.token_count = max(1, len(self.text) // 4)
        return self.token_count


@dataclass
class ContextBlock:
    """
    A block of context selected for inclusion in the prompt.

    Represents a segment of a Document that has been selected,
    potentially compressed, and scored for relevance.
    """
    document_id: str
    text: str
    relevance_score: float
    token_count: int

    # Provenance
    source_document: Optional[Document] = None
    compression_strategy: Optional[CompressionStrategy] = None

    # Quantum state information
    quantum_amplitude: complex = 0 + 0j
    measurement_probability: float = 0.0

    @property
    def information_density(self) -> float:
        """Bits of information per token."""
        return self.relevance_score / max(self.token_count, 1)


@dataclass
class QueryResult:
    """
    Result of a context extension query.

    Contains the selected context blocks and quality metrics.
    """
    query: str
    context_blocks: List[ContextBlock]
    total_tokens: int

    # Quality metrics (multi-dimensional evaluation)
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    coherence_score: float = 0.0
    accuracy_score: float = 0.0
    overall_quality: float = 0.0

    # Metadata
    retrieval_strategy: str = ""
    compression_used: Optional[CompressionStrategy] = None
    retrieval_time_ms: float = 0.0

    def compute_overall_quality(self) -> float:
        """
        Compute weighted quality score.

        Quality = 0.30·Relevance + 0.30·Completeness
                + 0.20·Coherence + 0.20·Accuracy
        """
        self.overall_quality = (
            0.30 * self.relevance_score +
            0.30 * self.completeness_score +
            0.20 * self.coherence_score +
            0.20 * self.accuracy_score
        )
        return self.overall_quality

    def format_context(self) -> str:
        """Format selected context blocks into a single string for the model."""
        parts = []
        for i, block in enumerate(self.context_blocks, 1):
            parts.append(f"[Context {i}] (relevance: {block.relevance_score:.3f})\n{block.text}")
        return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Model Client Protocol
# ──────────────────────────────────────────────────────────────────────────────

class ModelClient(Protocol):
    """
    Protocol for LLM model clients.

    Any model adapter must implement this interface to work with UCEF.
    """

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @property
    def context_window(self) -> int:
        """Return the native context window size in tokens."""
        ...

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the model."""
        ...

    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# 6. Model Profile
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    """
    Comprehensive model capability profile.

    Determines the optimal extension strategy based on measured capabilities.
    """
    model_name: str
    native_context_window: int
    context_category: ContextCategory

    # Measured capabilities
    performance_curve: Dict[int, float] = field(default_factory=dict)
    quality_retention: float = 0.0
    retrieval_capability: float = 0.0
    reasoning_strength: float = 0.0

    # Recommended configuration
    recommended_strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE
    optimal_compression_ratio: float = 0.3
    max_extended_context: int = 1_000_000

    def classify_context_category(self) -> ContextCategory:
        """Classify model into context category based on native window."""
        if self.native_context_window <= 32768:
            return ContextCategory.SMALL
        elif self.native_context_window <= 131072:
            return ContextCategory.MEDIUM
        elif self.native_context_window <= 262144:
            return ContextCategory.LARGE
        else:
            return ContextCategory.XLARGE


# ──────────────────────────────────────────────────────────────────────────────
# 7. Quality Issue Types
# ──────────────────────────────────────────────────────────────────────────────

class QualityIssueType(Enum):
    """Types of quality issues that can be detected."""
    MISSING_INFORMATION = "missing_information"
    NOISY_CONTEXT = "noisy_context"
    CONTRADICTORY_INFO = "contradictory_info"
    LOW_RELEVANCE = "low_relevance"
    INSUFFICIENT_DEPTH = "insufficient_depth"
    HALLUCINATION_RISK = "hallucination_risk"


@dataclass
class QualityIssue:
    """A detected quality issue in a model response."""
    issue_type: QualityIssueType
    severity: float  # 0.0 - 1.0
    description: str
    suggested_fix: str
    affected_context_ids: List[str] = field(default_factory=list)
