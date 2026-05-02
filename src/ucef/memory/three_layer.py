"""
Three-Layer Memory Manager — Orchestrator for Hot/Warm/Cold Storage

Coordinates document storage and retrieval across the three memory layers:
- Hot (Redis): <10ms, recent context, ~20K tokens
- Warm (ChromaDB): <100ms, semantic vectors, ~120K tokens
- Cold (filesystem): <500ms, full archive, unlimited

Migration policy:
- New documents → Cold (always) + Warm (if embeddings available)
- Accessed documents → promote to Hot
- Query results → promote to Hot
- Budget overflow → demote from Hot → Warm → Cold

References:
    - UCEF Architecture: docs/api/architecture.md
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ucef.core.config import MemorySystemConfig
from ucef.core.types import Document, HyperbolicPoint, TokenBudget
from ucef.memory.hot import RedisHotMemory
from ucef.memory.warm import ChromaWarmMemory
from ucef.memory.cold import FileSystemColdMemory


class ThreeLayerMemory:
    """
    Three-layer memory orchestrator.

    Manages document lifecycle across hot, warm, and cold storage layers,
    implementing automatic promotion/demotion based on access patterns
    and token budget constraints.

    Usage:
        memory = ThreeLayerMemory(config)
        await memory.store(documents, embeddings)
        results = await memory.retrieve(query_embedding, top_k=50)
    """

    def __init__(self, config: Optional[MemorySystemConfig] = None) -> None:
        self._config = config or MemorySystemConfig()
        self._hot = RedisHotMemory(self._config.hot)
        self._warm = ChromaWarmMemory(self._config.warm)
        self._cold = FileSystemColdMemory(self._config.cold)

    # ──────────────────────────────────────────────────────────────────────
    # Storage
    # ──────────────────────────────────────────────────────────────────────

    async def store(
        self,
        documents: Sequence[Document],
        embeddings: Optional[Sequence[NDArray[np.float64]]] = None,
    ) -> Dict[str, int]:
        """
        Store documents across memory layers.

        Distribution policy:
        - All documents → Cold (persistent archive)
        - Documents with embeddings → Warm (semantic index)
        - Recent/important documents → Hot (active cache)

        Args:
            documents: Documents to store.
            embeddings: Optional pre-computed embeddings for warm storage.

        Returns:
            Dict with counts per layer: {"hot": n, "warm": n, "cold": n}
        """
        counts = {"hot": 0, "warm": 0, "cold": 0}

        for i, doc in enumerate(documents):
            # Always store in cold (archive)
            if await self._cold.store(doc):
                counts["cold"] += 1

            # Store in warm if embedding is available
            emb = None
            if embeddings and i < len(embeddings):
                emb = embeddings[i]
            elif doc.euclidean_embedding is not None:
                emb = doc.euclidean_embedding

            if emb is not None:
                if await self._warm.store(doc, emb):
                    counts["warm"] += 1

        # Promote top documents to hot (by budget allocation)
        hot_budget = int(self._config.hot.max_tokens)
        hot_docs = self._select_for_hot(documents, hot_budget)

        for doc in hot_docs:
            if await self._hot.store(doc):
                counts["hot"] += 1

        return counts

    async def store_single(
        self,
        doc: Document,
        embedding: Optional[NDArray[np.float64]] = None,
    ) -> bool:
        """Store a single document across layers."""
        embs = [embedding] if embedding is not None else None
        result = await self.store([doc], embs)
        return result["cold"] > 0

    # ──────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query_embedding: NDArray[np.float64],
        top_k: int = 50,
    ) -> List[Tuple[Document, float]]:
        """
        Multi-layer retrieval with automatic promotion.

        Search order:
        1. Hot memory (fastest, recent context)
        2. Warm memory (semantic vectors)
        3. Cold not needed for normal retrieval

        Results are deduplicated and ranked by score.
        """
        results: Dict[str, Tuple[Document, float]] = {}

        # Layer 1: Hot memory (keyword-based fast check)
        hot_docs = await self._hot.retrieve_all()
        for doc in hot_docs:
            # Simple scoring: if doc is in hot memory, it's likely relevant
            results[doc.id] = (doc, 1.0)

        # Layer 2: Warm memory (semantic search)
        warm_results = await self._warm.retrieve(query_embedding, top_k=top_k)
        for doc, score in warm_results:
            if doc.id not in results:
                results[doc.id] = (doc, score)

            # Promote to hot if accessed
            await self._hot.store(doc)

        # Sort by score and return top_k
        sorted_results = sorted(
            results.values(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results[:top_k]

    async def retrieve_hyperbolic(
        self,
        query_point: HyperbolicPoint,
        top_k: int = 50,
    ) -> List[Tuple[Document, float]]:
        """Retrieve using hyperbolic geodesic distance."""
        return await self._warm.retrieve_hyperbolic(query_point, top_k)

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID, searching all layers.

        Search order: Hot → Warm → Cold.
        Promotes to Hot if found in lower layers.
        """
        # Hot
        doc = await self._hot.retrieve(doc_id)
        if doc is not None:
            return doc

        # Warm
        doc = await self._warm.get(doc_id)
        if doc is not None:
            await self._hot.store(doc)  # Promote
            return doc

        # Cold
        doc = await self._cold.retrieve(doc_id)
        if doc is not None:
            await self._hot.store(doc)  # Promote
            return doc

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Deletion
    # ──────────────────────────────────────────────────────────────────────

    async def delete(self, doc_id: str) -> bool:
        """Delete a document from all layers."""
        results = await asyncio.gather(
            self._hot.delete(doc_id),
            self._warm.delete(doc_id),
            self._cold.delete(doc_id),
        )
        return any(results)

    # ──────────────────────────────────────────────────────────────────────
    # Promotion / Demotion
    # ──────────────────────────────────────────────────────────────────────

    async def promote_to_hot(self, doc_id: str) -> bool:
        """Promote a document to hot memory from lower layers."""
        doc = await self.get_document(doc_id)
        if doc is None:
            return False
        return await self._hot.store(doc)

    # ──────────────────────────────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "hot": {
                "count": self._hot.count,
                "tokens": self._hot.current_tokens,
                "max_tokens": self._hot.max_tokens,
                "utilization": f"{self._hot.utilization:.1%}",
                "backend": "redis" if self._hot.using_redis else "in-memory",
            },
            "warm": {
                "count": self._warm.count,
                "backend": "chromadb" if self._warm.using_chroma else "in-memory",
            },
            "cold": {
                "count": await self._cold.count(),
                "format": self._cold._config.format,
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close all storage backends."""
        self._cold.close()

    def __del__(self) -> None:
        self.close()

    # ──────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _select_for_hot(
        documents: Sequence[Document],
        token_budget: int,
    ) -> List[Document]:
        """Select documents to promote to hot memory within budget."""
        selected: List[Document] = []
        total_tokens = 0

        for doc in documents:
            doc_tokens = doc.estimate_tokens()
            if total_tokens + doc_tokens <= token_budget:
                selected.append(doc)
                total_tokens += doc_tokens

        return selected
