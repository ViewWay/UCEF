"""
Warm Memory Layer — ChromaDB-backed Semantic Vector Store

The warm layer provides semantic retrieval over document embeddings.
It supports both Euclidean and hyperbolic embedding spaces and provides
< 100ms retrieval latency.

Design:
- Priority: medium (semantic relevance)
- Latency target: < 100ms
- Storage: persistent vector database
- Retrieval: cosine / L2 / hyperbolic nearest neighbor
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ucef.core.config import WarmMemoryConfig
from ucef.core.types import Document, HyperbolicPoint, poincare_distance


class ChromaWarmMemory:
    """
    Warm memory layer using ChromaDB for semantic vector storage.

    In production, wraps a real ChromaDB client. Falls back to
    in-memory numpy arrays with brute-force search for development.

    Supports:
    - Semantic search via embedding similarity
    - Metadata filtering
    - Hyperbolic distance search (via post-filtering)
    - Document CRUD operations
    """

    def __init__(self, config: Optional[WarmMemoryConfig] = None) -> None:
        self._config = config or WarmMemoryConfig()
        self._chroma_client: Any = None
        self._collection: Any = None

        # Fallback in-memory storage
        self._documents: Dict[str, Document] = {}
        self._embeddings: Dict[str, NDArray[np.float64]] = {}
        self._hyperbolic_points: Dict[str, HyperbolicPoint] = {}

        if self._config.enabled:
            self._init_chroma()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB client. Falls back to in-memory on failure."""
        try:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(
                path=self._config.persist_directory,
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=self._config.collection_name,
                metadata={"hnsw:space": self._config.distance_metric},
            )
        except Exception:
            self._chroma_client = None
            self._collection = None

    # ──────────────────────────────────────────────────────────────────────
    # Core Operations
    # ──────────────────────────────────────────────────────────────────────

    async def store(self, doc: Document, embedding: NDArray[np.float64]) -> bool:
        """
        Store a document with its embedding.

        Args:
            doc: Document to store.
            embedding: Euclidean embedding vector.

        Returns:
            True if stored successfully.
        """
        if not self._config.enabled:
            return False

        self._documents[doc.id] = doc
        self._embeddings[doc.id] = embedding.astype(np.float64)

        if doc.hyperbolic_embedding is not None:
            self._hyperbolic_points[doc.id] = doc.hyperbolic_embedding

        if self._collection is not None:
            return self._store_chroma(doc, embedding)

        return True

    async def store_batch(
        self,
        docs: Sequence[Document],
        embeddings: Sequence[NDArray[np.float64]],
    ) -> int:
        """Store multiple documents with embeddings."""
        stored = 0
        for doc, emb in zip(docs, embeddings):
            if await self.store(doc, emb):
                stored += 1
        return stored

    async def retrieve(
        self,
        query_embedding: NDArray[np.float64],
        top_k: int = 50,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents by embedding similarity.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results.
            metadata_filter: Optional metadata filter.

        Returns:
            List of (document, similarity_score) tuples.
        """
        if not self._config.enabled:
            return []

        if self._collection is not None:
            return self._retrieve_chroma(query_embedding, top_k, metadata_filter)
        else:
            return self._retrieve_fallback(query_embedding, top_k)

    async def retrieve_hyperbolic(
        self,
        query_point: HyperbolicPoint,
        top_k: int = 50,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents by hyperbolic geodesic distance.

        Only considers documents that have hyperbolic embeddings.
        Uses brute-force search (HNSW in hyperbolic space is Phase 3).

        Args:
            query_point: Query point in Poincaré ball.
            top_k: Number of results.

        Returns:
            List of (document, geodesic_distance) tuples, ascending.
        """
        if not self._hyperbolic_points:
            return []

        scored: List[Tuple[str, float]] = []
        for doc_id, point in self._hyperbolic_points.items():
            dist = poincare_distance(query_point, point)
            scored.append((doc_id, dist))

        scored.sort(key=lambda x: x[1])

        results: List[Tuple[Document, float]] = []
        for doc_id, dist in scored[:top_k]:
            if doc_id in self._documents:
                results.append((self._documents[doc_id], dist))

        return results

    async def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        removed = False

        if doc_id in self._documents:
            del self._documents[doc_id]
            removed = True
        if doc_id in self._embeddings:
            del self._embeddings[doc_id]
        if doc_id in self._hyperbolic_points:
            del self._hyperbolic_points[doc_id]

        if self._collection is not None:
            try:
                self._collection.delete(ids=[doc_id])
            except Exception:
                pass

        return removed

    async def get(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._documents.get(doc_id)

    # ──────────────────────────────────────────────────────────────────────
    # Fallback (In-Memory) Implementation
    # ──────────────────────────────────────────────────────────────────────

    def _retrieve_fallback(
        self,
        query_embedding: NDArray[np.float64],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        """Brute-force cosine similarity search."""
        if not self._embeddings:
            return []

        query_norm = np.linalg.norm(query_embedding)
        if query_norm < 1e-10:
            return []

        query_unit = query_embedding / query_norm

        scored: List[Tuple[str, float]] = []
        for doc_id, emb in self._embeddings.items():
            emb_norm = np.linalg.norm(emb)
            if emb_norm < 1e-10:
                continue
            similarity = float(np.dot(query_unit, emb / emb_norm))
            scored.append((doc_id, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[Tuple[Document, float]] = []
        for doc_id, score in scored[:top_k]:
            if doc_id in self._documents:
                results.append((self._documents[doc_id], score))

        return results

    # ──────────────────────────────────────────────────────────────────────
    # ChromaDB Implementation
    # ──────────────────────────────────────────────────────────────────────

    def _store_chroma(self, doc: Document, embedding: NDArray[np.float64]) -> bool:
        """Store in ChromaDB collection."""
        try:
            self._collection.upsert(
                ids=[doc.id],
                embeddings=[embedding.tolist()],
                documents=[doc.text],
                metadatas=[doc.metadata if doc.metadata else None],
            )
            return True
        except Exception:
            return False

    def _retrieve_chroma(
        self,
        query_embedding: NDArray[np.float64],
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Tuple[Document, float]]:
        """Retrieve from ChromaDB collection."""
        try:
            kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": top_k,
            }
            if metadata_filter:
                kwargs["where"] = metadata_filter

            results = self._collection.query(**kwargs)

            documents: List[Tuple[Document, float]] = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    text = results["documents"][0][i] if results["documents"] else ""
                    score = float(results["distances"][0][i]) if results["distances"] else 0.0
                    metadata = (
                        results["metadatas"][0][i]
                        if results["metadatas"] and results["metadatas"][0]
                        else {}
                    )
                    documents.append((
                        Document(id=doc_id, text=text, metadata=metadata),
                        score,
                    ))
            return documents
        except Exception:
            return []

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of documents stored."""
        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                pass
        return len(self._documents)

    @property
    def using_chroma(self) -> bool:
        return self._collection is not None

    def get_embedding(self, doc_id: str) -> Optional[NDArray[np.float64]]:
        """Get the Euclidean embedding for a document."""
        return self._embeddings.get(doc_id)
