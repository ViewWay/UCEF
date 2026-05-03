"""Tests for retrieval engine: hyperbolic, quantum, fusion."""
import numpy as np

from ucef.core.types import HyperbolicPoint, Document, ContextBlock, TokenBudget
from ucef.retrieval.hyperbolic import HyperbolicRetriever
from ucef.retrieval.quantum import QuantumSelector
from ucef.retrieval.fusion import ReciprocalRankFusion, WeightedScoreFusion, HybridFusion


def make_block(doc_id, text, score=0.5):
    return ContextBlock(
        document_id=doc_id, text=text,
        relevance_score=score, token_count=len(text.split()),
    )


class TestHyperbolicRetriever:
    def setup_method(self):
        self.retriever = HyperbolicRetriever()

    def test_add_and_count(self):
        docs = [
            Document(id=f"doc{i}", text=f"Document number {i}")
            for i in range(20)
        ]
        for doc in docs:
            self.retriever.add_document(doc)
        # n_indexed should be > 0 after adding docs
        assert self.retriever.n_indexed > 0

    def test_retrieve_by_text(self):
        docs = [
            Document(id=f"doc{i}", text=f"Document number {i}")
            for i in range(20)
        ]
        for doc in docs:
            self.retriever.add_document(doc)
        results = self.retriever.retrieve_by_text("Document number 5", top_k=5)
        assert len(results) <= 5
        assert len(results) > 0

    def test_retrieve_results_have_scores(self):
        docs = [
            Document(id=f"doc{i}", text=f"Document number {i}")
            for i in range(10)
        ]
        for doc in docs:
            self.retriever.add_document(doc)
        results = self.retriever.retrieve_by_text("Document", top_k=3)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_empty_retriever_search(self):
        results = self.retriever.retrieve_by_text("query", top_k=5)
        assert results == []

    def test_remove_document(self):
        doc = Document(id="removable", text="Remove me")
        self.retriever.add_document(doc)
        assert self.retriever.n_indexed == 1
        self.retriever.remove_document("removable")
        assert self.retriever.n_indexed == 0

    def test_get_embedding(self):
        doc = Document(id="embed_test", text="Embedding test document")
        self.retriever.add_document(doc)
        emb = self.retriever.get_embedding("embed_test")
        assert emb is not None
        assert isinstance(emb, HyperbolicPoint)


class TestQuantumSelector:
    def setup_method(self):
        self.selector = QuantumSelector()

    def test_select_from_candidates(self):
        docs_with_scores = [
            (Document(id=f"blk{i}", text=f"Block {i}"), 1.0 - i * 0.1)
            for i in range(10)
        ]
        budget = TokenBudget(total=10000, retrieved_context=5000)
        selected = self.selector.select(docs_with_scores, budget=budget)
        assert len(selected) > 0

    def test_select_with_query_weights(self):
        docs_with_scores = [
            (Document(id="high", text="High relevance"), 0.95),
            (Document(id="mid", text="Medium relevance"), 0.5),
            (Document(id="low", text="Low relevance"), 0.1),
        ]
        query_weights = np.random.randn(8)
        budget = TokenBudget(total=10000, retrieved_context=5000)
        selected = self.selector.select(docs_with_scores, query_weights=query_weights, budget=budget)
        assert len(selected) > 0
        ids = [b.document_id for b in selected]
        assert "high" in ids

    def test_select_empty_candidates(self):
        budget = TokenBudget(total=10000)
        selected = self.selector.select([], budget=budget)
        assert selected == []


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        d_a, d_b, d_c = Document(id="a",text="a"), Document(id="b",text="b"), Document(id="c",text="c")
        rankings = [
            [(d_a, 0.9), (d_b, 0.8), (d_c, 0.7)],
            [(d_b, 0.95), (d_c, 0.85), (d_a, 0.75)],
        ]
        rrf = ReciprocalRankFusion(k=60)
        result = rrf.fuse(rankings)
        assert len(result) == 3
        result_ids = {doc.id for doc, score in result}
        assert result_ids == {"a", "b", "c"}

    def test_empty_rankings(self):
        rrf = ReciprocalRankFusion()
        result = rrf.fuse([])
        assert result == []


class TestWeightedScoreFusion:
    def test_weighted_fusion(self):
        d_a = Document(id="a", text="a")
        d_b = Document(id="b", text="b")
        score_lists = [
            [(d_a, 0.9), (d_b, 0.5)],
            [(d_a, 0.7), (d_b, 0.8)],
        ]
        ws = WeightedScoreFusion(weights=[0.6, 0.4])
        result = ws.fuse(score_lists)
        assert len(result) == 2
        assert result[0][0].id == "a"


class TestHybridFusion:
    def test_hybrid_combines_methods(self):
        d_a, d_b, d_c = Document(id="a",text="a"), Document(id="b",text="b"), Document(id="c",text="c")
        rankings = [
            [(d_a, 0.9), (d_b, 0.8), (d_c, 0.7)],
            [(d_b, 0.95), (d_c, 0.85), (d_a, 0.75)],
        ]
        hf = HybridFusion(alpha=0.5, rrf_k=60)
        result = hf.fuse(rankings)
        assert len(result) == 3
