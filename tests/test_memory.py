"""Tests for memory system: hot, warm, cold."""
import asyncio
import numpy as np

from ucef.core.types import Document
from ucef.memory.hot import RedisHotMemory
from ucef.memory.warm import ChromaWarmMemory
from ucef.memory.cold import FileSystemColdMemory


def make_doc(doc_id, text):
    return Document(id=doc_id, text=text)


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestHotMemory:
    def setup_method(self):
        self.mem = RedisHotMemory()

    def test_store_and_retrieve(self):
        doc = make_doc("h1", "Hello from hot memory")
        run_async(self.mem.store(doc))
        result = run_async(self.mem.retrieve("h1"))
        assert result is not None
        assert result.id == "h1"

    def test_retrieve_nonexistent(self):
        result = run_async(self.mem.retrieve("nope"))
        assert result is None

    def test_delete(self):
        doc = make_doc("h2", "Delete me")
        run_async(self.mem.store(doc))
        assert self.mem.count >= 1
        run_async(self.mem.delete("h2"))
        result = run_async(self.mem.retrieve("h2"))
        assert result is None

    def test_count(self):
        doc = make_doc("h3", "Stats test")
        run_async(self.mem.store(doc))
        assert self.mem.count >= 1

    def test_utilization(self):
        doc = make_doc("h4", "Util test")
        run_async(self.mem.store(doc))
        assert self.mem.utilization >= 0


class TestWarmMemory:
    def setup_method(self):
        self.mem = ChromaWarmMemory()

    def test_store_and_get(self):
        doc = make_doc("w1", "Hello from warm memory")
        emb = np.random.randn(8).astype(np.float64)
        emb = emb / (np.linalg.norm(emb) + 1e-8) * 0.5
        run_async(self.mem.store(doc, emb))
        result = run_async(self.mem.get("w1"))
        assert result is not None
        assert result.id == "w1"

    def test_retrieve_by_embedding(self):
        for i in range(5):
            doc = make_doc(f"w{i}", f"Document about topic {i}")
            emb = np.random.randn(8).astype(np.float64)
            emb = emb / (np.linalg.norm(emb) + 1e-8) * 0.5
            run_async(self.mem.store(doc, emb))
        query_emb = np.random.randn(8).astype(np.float64)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8) * 0.5
        results = run_async(self.mem.retrieve(query_emb, top_k=3))
        assert len(results) > 0

    def test_delete(self):
        doc = make_doc("wd", "Delete me warm")
        emb = np.random.randn(8).astype(np.float64) * 0.3
        run_async(self.mem.store(doc, emb))
        run_async(self.mem.delete("wd"))
        result = run_async(self.mem.get("wd"))
        assert result is None


class TestColdMemory:
    def setup_method(self):
        self.mem = FileSystemColdMemory()

    def test_store_and_retrieve(self):
        doc = make_doc("c1", "Cold storage document")
        run_async(self.mem.store(doc))
        result = run_async(self.mem.retrieve("c1"))
        assert result is not None
        assert result.id == "c1"

    def test_delete(self):
        doc = make_doc("c3", "To be deleted")
        run_async(self.mem.store(doc))
        run_async(self.mem.delete("c3"))
        result = run_async(self.mem.retrieve("c3"))
        assert result is None

    def test_list_ids(self):
        run_async(self.mem.store(make_doc("cl1", "Cold 1")))
        run_async(self.mem.store(make_doc("cl2", "Cold 2")))
        ids = run_async(self.mem.list_ids())
        assert "cl1" in ids
        assert "cl2" in ids
