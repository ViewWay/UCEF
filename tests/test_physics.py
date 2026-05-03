"""Tests for physics-inspired models: thermodynamic and quantum field."""
import numpy as np

from ucef.core.types import ContextBlock
from ucef.physics.thermodynamic import ThermodynamicModel
from ucef.physics.quantum_field import RenormalizationGroup


def make_blocks(n=15):
    return [
        ContextBlock(
            document_id=f"ph{i}",
            text=f"Physics block {i} with content about science. " * 5,
            relevance_score=0.5 + 0.5 * np.random.random(),
            token_count=20 + i,
        )
        for i in range(n)
    ]


class TestThermodynamicModel:
    def setup_method(self):
        self.model = ThermodynamicModel()

    def test_free_energy(self):
        f = self.model.free_energy(energy=0.5, entropy=0.3)
        assert isinstance(f, float)

    def test_energy(self):
        blocks = make_blocks(10)
        e = self.model.energy(blocks[0], query_words={"physics", "science"})
        assert isinstance(e, float)

    def test_entropy_contribution(self):
        blocks = make_blocks(10)
        s = self.model.entropy_contribution(blocks[0], already_selected_words={"physics"})
        assert isinstance(s, float)

    def test_boltzmann_probabilities(self):
        blocks = make_blocks(5)
        probs = self.model.boltzmann_probabilities(blocks, query_words={"physics"})
        assert len(probs) == len(blocks)
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_anneal(self):
        self.model.anneal()
        assert self.model._temperature < 1.0

    def test_reset(self):
        self.model.anneal()
        self.model.reset()
        assert self.model._temperature == 1.0


class TestRenormalizationGroup:
    def setup_method(self):
        self.rg = RenormalizationGroup()

    def test_coarse_grain(self):
        text = "This is a long text for coarse graining. " * 20
        result = self.rg.coarse_grain(text, target_ratio=0.5)
        assert isinstance(result, str)
        assert len(result) < len(text)

    def test_relevance_flow(self):
        blocks = make_blocks(10)
        flow = self.rg.relevance_flow(blocks, query="physics")
        assert len(flow) == len(blocks)
        assert all(isinstance(f, float) for f in flow)
