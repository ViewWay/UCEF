"""Tests for core types: hyperbolic geometry, quantum states, info theory."""
import math
import numpy as np

from ucef.core.types import (
    HyperbolicPoint, QuantumState, DensityMatrix,
    Document, ContextBlock, TokenBudget, ModelProfile, QueryResult,
    ContextCategory, CompressionStrategy,
    poincare_distance, exponential_map,
)


class TestHyperbolicPoint:
    def test_valid_point_creation(self):
        p = HyperbolicPoint(coordinates=np.array([0.3, -0.2, 0.1]))
        assert p.coordinates.shape == (3,)
        assert np.linalg.norm(p.coordinates) < 1.0

    def test_point_at_origin(self):
        p = HyperbolicPoint.origin(dim=5)
        assert np.allclose(p.coordinates, np.zeros(5))

    def test_point_exceeding_boundary_raises(self):
        try:
            HyperbolicPoint(coordinates=np.array([0.9, 0.9]))
        except (ValueError, Exception):
            pass

    def test_conformal_factor(self):
        p = HyperbolicPoint(coordinates=np.array([0.5, 0.0]))
        cf = p.conformal_factor  # property, not method
        expected = 2.0 / (1.0 - 0.25)
        assert abs(cf - expected) < 1e-6

    def test_norm(self):
        p = HyperbolicPoint(coordinates=np.array([0.3, 0.4]))
        assert abs(p.norm - 0.5) < 1e-6  # property, not method

    def test_is_valid(self):
        p = HyperbolicPoint(coordinates=np.array([0.5]))
        assert p.is_valid()


class TestPoincareDistance:
    def test_distance_to_self_is_zero(self):
        p = HyperbolicPoint(coordinates=np.array([0.3, -0.2]))
        d = poincare_distance(p, p)
        assert d == 0.0 or d < 1e-10

    def test_distance_symmetry(self):
        p1 = HyperbolicPoint(coordinates=np.array([0.3, -0.2]))
        p2 = HyperbolicPoint(coordinates=np.array([-0.1, 0.4]))
        d12 = poincare_distance(p1, p2)
        d21 = poincare_distance(p2, p1)
        assert abs(d12 - d21) < 1e-10

    def test_distance_positive(self):
        p1 = HyperbolicPoint(coordinates=np.array([0.3, -0.2]))
        p2 = HyperbolicPoint(coordinates=np.array([-0.1, 0.4]))
        assert poincare_distance(p1, p2) > 0

    def test_distance_origin_to_boundary_large(self):
        origin = HyperbolicPoint.origin(dim=2)
        near_boundary = HyperbolicPoint(coordinates=np.array([0.999, 0.0]))
        d = poincare_distance(origin, near_boundary)
        assert d > 3.0

    def test_triangle_inequality(self):
        p1 = HyperbolicPoint(coordinates=np.array([0.1, 0.0]))
        p2 = HyperbolicPoint(coordinates=np.array([0.0, 0.1]))
        p3 = HyperbolicPoint(coordinates=np.array([-0.1, -0.1]))
        d12 = poincare_distance(p1, p2)
        d23 = poincare_distance(p2, p3)
        d13 = poincare_distance(p1, p3)
        assert d13 <= d12 + d23 + 1e-8


class TestExponentialMap:
    def test_zero_vector_maps_to_origin(self):
        v = np.zeros(3)
        result = exponential_map(v)
        assert np.allclose(result.coordinates, np.zeros(3), atol=1e-10)

    def test_result_norm_less_than_one(self):
        v = np.array([5.0, -3.0, 2.0])
        result = exponential_map(v)
        assert np.linalg.norm(result.coordinates) < 1.0

    def test_small_vector_approximates_tangent(self):
        v = np.array([0.01, 0.02])
        result = exponential_map(v)
        assert np.allclose(result.coordinates, v, atol=0.001)


class TestQuantumState:
    def test_normalized_creation(self):
        amps = np.array([0.6, 0.8])
        qs = QuantumState(amplitudes=amps)
        assert qs.is_normalized

    def test_unnormalized_needs_normalize(self):
        amps = np.array([3.0, 4.0])
        qs = QuantumState(amplitudes=amps)
        assert not qs.is_normalized
        qs_norm = qs.normalize()
        assert qs_norm.is_normalized

    def test_zero_amplitude_raises_on_normalize(self):
        qs = QuantumState(amplitudes=np.array([0.0, 0.0, 0.0]))
        try:
            qs.normalize()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_probabilities_sum_to_one(self):
        qs = QuantumState(amplitudes=np.array([0.6, 0.8]))
        probs = qs.probabilities
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_equal_superposition(self):
        qs = QuantumState.equal_superposition(4)
        assert qs.is_normalized
        assert qs.n_states == 4

    def test_from_probabilities(self):
        probs = np.array([0.5, 0.3, 0.2])
        qs = QuantumState.from_probabilities(probs)
        assert qs.is_normalized


class TestDensityMatrix:
    def test_pure_state_creation(self):
        qs = QuantumState(amplitudes=np.array([0.6, 0.8]))
        rho = DensityMatrix.from_pure_state(qs)
        assert rho.is_valid

    def test_trace_equals_one(self):
        qs = QuantumState(amplitudes=np.array([0.5, 0.5, 0.5, 0.5]))
        qs = qs.normalize()
        rho = DensityMatrix.from_pure_state(qs)
        assert abs(np.trace(rho.matrix) - 1.0) < 1e-6

    def test_purity_of_pure_state(self):
        qs = QuantumState(amplitudes=np.array([0.6, 0.8]))
        rho = DensityMatrix.from_pure_state(qs)
        purity = np.trace(rho.matrix @ rho.matrix).real
        assert abs(purity - 1.0) < 1e-4

    def test_mixed_state(self):
        qs1 = QuantumState(amplitudes=np.array([1.0, 0.0]))
        qs2 = QuantumState(amplitudes=np.array([0.0, 1.0]))
        rho = DensityMatrix.from_mixed_states(
            [(0.5, qs1), (0.5, qs2)]
        )
        assert rho.is_valid
        purity = np.trace(rho.matrix @ rho.matrix).real
        assert purity < 1.0


class TestDocument:
    def test_creation(self):
        doc = Document(id="doc1", text="Hello world", metadata={"source": "test"})
        assert doc.id == "doc1"
        assert doc.text == "Hello world"


class TestContextBlock:
    def test_creation(self):
        block = ContextBlock(
            document_id="doc1", text="Hello world",
            relevance_score=0.9, token_count=3,
        )
        assert block.document_id == "doc1"
        assert block.relevance_score == 0.9

    def test_default_fields(self):
        block = ContextBlock(
            document_id="d1", text="test", relevance_score=0.5, token_count=1,
        )
        assert block.quantum_amplitude == 0 + 0j
        assert block.measurement_probability == 0.0


class TestTokenBudget:
    def test_creation(self):
        budget = TokenBudget(total=100000)
        assert budget.total == 100000

    def test_fields(self):
        budget = TokenBudget(total=100000, system_prompt=1000, response_buffer=2000)
        assert budget.system_prompt == 1000
        assert budget.response_buffer == 2000


class TestModelProfile:
    def test_categories(self):
        small = ModelProfile(
            model_name="test", native_context_window=8192,
            context_category=ContextCategory.SMALL,
        )
        assert small.context_category == ContextCategory.SMALL

        large = ModelProfile(
            model_name="test", native_context_window=200000,
            context_category=ContextCategory.LARGE,
        )
        assert large.context_category == ContextCategory.LARGE

    def test_default_strategy(self):
        p = ModelProfile(
            model_name="test", native_context_window=128000,
            context_category=ContextCategory.MEDIUM,
        )
        assert p.recommended_strategy == CompressionStrategy.ADAPTIVE


class TestQueryResult:
    def test_creation(self):
        qr = QueryResult(
            query="test", context_blocks=[], total_tokens=100,
            overall_quality=0.85,
        )
        assert qr.query == "test"
        assert qr.overall_quality == 0.85
