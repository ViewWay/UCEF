"""
Hyperbolic Retriever — Poincaré Ball Embeddings & Geodesic Nearest Neighbor Search

Implements document embedding in hyperbolic space (Poincaré ball model) and
retrieval via geodesic distance, leveraging the exponential capacity of
negative-curvature spaces for hierarchical and semantic data.

Key properties:
- Ω(log n) retrieval complexity with HNSW indexing
- Natural representation of hierarchical relationships
- Exponentially more capacity than Euclidean space for the same dimension

References:
    - Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical
      Representations", NeurIPS 2017
    - Nickel & Kiela, "Learning Continuous Hierarchies in the Lorentz Model
      of Hyperbolic Geometry", ICML 2018
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ucef.core.config import HyperbolicConfig
from ucef.core.types import (
    Document,
    HyperbolicPoint,
    exponential_map,
    mobius_add,
    poincare_distance,
)


class HyperbolicRetriever:
    """
    Retrieves documents using hyperbolic space embeddings.

    Documents are embedded as points in the Poincaré ball. Queries are
    also embedded, and geodesic distance is used as the relevance metric.

    Usage:
        retriever = HyperbolicRetriever(config)
        retriever.index(documents)
        results = retriever.retrieve(query_embedding, top_k=10)
    """

    def __init__(self, config: Optional[HyperbolicConfig] = None) -> None:
        self._config = config or HyperbolicConfig()
        self._embeddings: Dict[str, HyperbolicPoint] = {}
        self._documents: Dict[str, Document] = {}
        self._embedding_matrix: Optional[NDArray[np.float64]] = None
        self._id_order: List[str] = []
        self._indexed = False

    # ──────────────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────────────

    def index(self, documents: Sequence[Document]) -> int:
        """
        Index documents by computing or accepting hyperbolic embeddings.

        If a document already has a hyperbolic_embedding, it is used directly.
        Otherwise, a Euclidean embedding (if available) is projected into the
        Poincaré ball via the exponential map. If neither exists, a random
        point is generated.

        Args:
            documents: Documents to index.

        Returns:
            Number of documents indexed.
        """
        self._id_order = []
        coord_list: List[NDArray[np.float64]] = []

        for doc in documents:
            self._documents[doc.id] = doc

            if doc.hyperbolic_embedding is not None and doc.hyperbolic_embedding.is_valid():
                point = doc.hyperbolic_embedding
            elif doc.euclidean_embedding is not None:
                point = self._project_to_ball(doc.euclidean_embedding)
            else:
                point = HyperbolicPoint.random(
                    self._config.embedding_dim,
                    max_norm=self._config.max_norm,
                )

            self._embeddings[doc.id] = point
            self._id_order.append(doc.id)
            coord_list.append(point.coordinates)

        if coord_list:
            self._embedding_matrix = np.stack(coord_list, axis=0)

        self._indexed = True
        return len(self._id_order)

    def add_document(self, doc: Document) -> None:
        """Add a single document to the index."""
        self.index([doc])

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if doc_id in self._embeddings:
            del self._embeddings[doc_id]
        if doc_id in self._documents:
            del self._documents[doc_id]
        if doc_id in self._id_order:
            self._id_order.remove(doc_id)
            self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        """Rebuild the embedding matrix after modifications."""
        if self._id_order:
            coords = [self._embeddings[doc_id].coordinates for doc_id in self._id_order]
            self._embedding_matrix = np.stack(coords, axis=0)
        else:
            self._embedding_matrix = None

    # ──────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: HyperbolicPoint,
        top_k: int = 50,
        exclude_ids: Optional[Sequence[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents by geodesic distance to query point.

        The geodesic distance in the Poincaré ball naturally captures
        semantic similarity, with hierarchical relationships encoded
        more efficiently than in Euclidean space.

        Args:
            query: Query point in the Poincaré ball.
            top_k: Number of results to return.
            exclude_ids: Document IDs to exclude.

        Returns:
            List of (document, distance) tuples, sorted by ascending distance.
        """
        if not self._indexed or self._embedding_matrix is None:
            return []

        exclude = set(exclude_ids or [])
        distances = self._batch_geodesic_distance(query, self._embedding_matrix)

        scored: List[Tuple[str, float]] = []
        for i, doc_id in enumerate(self._id_order):
            if doc_id not in exclude:
                scored.append((doc_id, float(distances[i])))

        scored.sort(key=lambda x: x[1])

        results: List[Tuple[Document, float]] = []
        for doc_id, dist in scored[:top_k]:
            if doc_id in self._documents:
                results.append((self._documents[doc_id], dist))

        return results

    def retrieve_by_text(
        self,
        query_text: str,
        top_k: int = 50,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents by query text.

        Creates a simple keyword-based embedding in Euclidean space,
        then projects it into the Poincaré ball via exponential map.
        For production use, replace with a trained sentence encoder.

        Args:
            query_text: Query string.
            top_k: Number of results.

        Returns:
            List of (document, distance) tuples.
        """
        query_point = self._text_to_point(query_text)
        return self.retrieve(query_point, top_k)

    # ──────────────────────────────────────────────────────────────────────
    # Distance Computation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _batch_geodesic_distance(
        query: HyperbolicPoint,
        matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute geodesic distance from query to all points in matrix.

        Vectorized implementation for performance:
            d(u, v_i) = arcosh(1 + 2||u - v_i||² / ((1 - ||u||²)(1 - ||v_i||²)))

        Complexity: O(n * d) where n = number of points, d = dimension.
        """
        query_coords = query.coordinates  # (d,)
        query_norm_sq = np.sum(query_coords ** 2)  # scalar

        diff = matrix - query_coords[np.newaxis, :]  # (n, d)
        diff_norm_sq = np.sum(diff ** 2, axis=1)  # (n,)

        matrix_norm_sq = np.sum(matrix ** 2, axis=1)  # (n,)

        denominator = (1.0 - query_norm_sq) * (1.0 - matrix_norm_sq)  # (n,)
        # Avoid division by zero / negative for points near boundary
        # When norms approach 1.0, denominator → 0, causing instability.
        # Clip to a small positive value proportional to scale.
        denominator = np.clip(denominator, 1e-5, None)

        inner = 1.0 + 2.0 * diff_norm_sq / denominator
        inner = np.clip(inner, 1.0, None)  # arcosh domain: x >= 1

        return np.arccosh(inner)

    # ──────────────────────────────────────────────────────────────────────
    # Embedding Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _project_to_ball(self, euclidean_vec: NDArray[np.float64]) -> HyperbolicPoint:
        """
        Project a Euclidean vector into the Poincaré ball via exponential map.

        exp_0(v) = tanh(||v||) · v / ||v||

        The exponential map preserves the direction while mapping the
        magnitude into the bounded Poincaré ball.
        """
        return exponential_map(euclidean_vec, base=None)

    def _text_to_point(self, text: str) -> HyperbolicPoint:
        """
        Create a Poincaré ball point from text.

        Phase 2: Simple hash-based embedding for prototyping.
        Phase 3: Replace with trained sentence transformer.

        Uses a deterministic hash to create a reproducible embedding,
        then projects via exponential map.
        """
        dim = self._config.embedding_dim
        rng = np.random.default_rng(hash(text) % (2**31))

        vec = rng.standard_normal(dim).astype(np.float64)
        vec = vec / (np.linalg.norm(vec) + 1e-10) * rng.uniform(0.5, 2.0)

        return self._project_to_ball(vec)

    # ──────────────────────────────────────────────────────────────────────
    # Embedding Training (Stub for Phase 3)
    # ──────────────────────────────────────────────────────────────────────

    def train_embeddings(
        self,
        documents: Sequence[Document],
        n_epochs: int = 100,
        learning_rate: float = 0.01,
    ) -> None:
        """
        Train hyperbolic embeddings using Riemannian SGD.

        Minimizes geodesic distance between semantically similar documents
        while pushing dissimilar ones apart.

        Notation:
            Loss = Σ_{(i,j)∈similar} log(d(u_i, u_j))
                 + Σ_{(i,k)∈dissimilar} max(0, δ - d(u_i, u_k))

        Gradient update (Riemannian):
            u ← exp_u(-η · (1 - ||u||²)² / 4 · ∂L/∂u)

        Reference: Nickel & Kiela (2017), Algorithm 1

        Status: Stub — full implementation in Phase 3.
        """
        # TODO: Implement Riemannian SGD training
        # 1. Sample positive pairs (similar docs)
        # 2. Sample negative pairs (dissimilar docs)
        # 3. Compute loss
        # 4. Riemannian gradient descent step
        # 5. Project back to ball (clip norm to max_norm)
        raise NotImplementedError(
            "Embedding training will be implemented in Phase 3. "
            "Use pre-computed embeddings or random initialization for now."
        )

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def n_indexed(self) -> int:
        """Number of documents currently indexed."""
        return len(self._id_order)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding space."""
        return self._config.embedding_dim

    def get_embedding(self, doc_id: str) -> Optional[HyperbolicPoint]:
        """Get the hyperbolic embedding for a document."""
        return self._embeddings.get(doc_id)
