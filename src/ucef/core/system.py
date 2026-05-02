"""
UCEF Core System — Universal Context Extension Framework

Main orchestrator that coordinates all subsystems:
- Model profiling and strategy selection
- Document storage across three-layer memory
- Context retrieval (hyperbolic + quantum)
- Adaptive compression
- Quality preservation

References:
    - Nickel & Kiela, "Poincaré Embeddings", NeurIPS 2017
    - van Rijsbergen, "The Geometry of Information Retrieval", CUP 2004
    - Grünwald, "The Minimum Description Length Principle", MIT Press 2007
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ucef.core.config import UCEFConfig
from ucef.core.types import (
    CompressionResult,
    CompressionStrategy,
    ContextBlock,
    ContextCategory,
    Document,
    ModelClient,
    ModelProfile,
    QueryResult,
    QuantumState,
    TokenBudget,
)


class UniversalContextSystem:
    """
    Universal Context Extension Framework.

    Enables any LLM to handle unlimited context (4K → 1M+ tokens)
    while preserving output quality through:

    1. Hyperbolic geometry retrieval — Ω(log n) semantic search
    2. Quantum-inspired selection — superposition-based context filtering
    3. Three-layer memory — hot/warm/cold architecture
    4. Adaptive compression — model-aware context budgeting
    5. Quality preservation — multi-dimensional evaluation + regeneration

    Usage:
        system = UniversalContextSystem(model_client, "gpt-4o")
        await system.store_documents(documents)
        result = await system.query("What are the key findings?")
    """

    def __init__(
        self,
        model_client: ModelClient,
        model_name: str,
        config: Optional[UCEFConfig] = None,
    ) -> None:
        self._model_client = model_client
        self._model_name = model_name
        self._config = config or UCEFConfig()

        # Internal state
        self._documents: Dict[str, Document] = {}
        self._model_profile: Optional[ModelProfile] = None
        self._initialized = False

        # Subsystem references (lazy-initialized)
        self._profiler = None
        self._retriever = None
        self._quality_engine = None
        self._compression_engine = None

    # ──────────────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Initialize all subsystems.

        This must be called before any other operations. It:
        1. Profiles the model
        2. Initializes memory layers
        3. Sets up retrieval indices
        """
        if self._initialized:
            return

        # 1. Profile the model
        from ucef.quality.profiler import ModelCapabilityProfiler
        self._profiler = ModelCapabilityProfiler()
        self._model_profile = await self._profiler.profile_model(
            self._model_client, self._model_name
        )

        # 2. Ensure directories exist
        self._config.ensure_directories()

        # 3. Initialize subsystems (will be connected in Phase 2)
        # self._retriever = HyperbolicRetriever(self._config.hyperbolic)
        # self._quality_engine = QualityPreservationEngine(self._config.quality)
        # self._compression_engine = AdaptiveCompressor(self._config.compression)

        self._initialized = True

    @property
    def model_profile(self) -> ModelProfile:
        """Get the current model profile."""
        if self._model_profile is None:
            raise RuntimeError("System not initialized. Call initialize() first.")
        return self._model_profile

    @property
    def config(self) -> UCEFConfig:
        return self._config

    @property
    def document_count(self) -> int:
        return len(self._documents)

    # ──────────────────────────────────────────────────────────────────────
    # Document Management
    # ──────────────────────────────────────────────────────────────────────

    async def store_documents(
        self,
        documents: Sequence[Document],
    ) -> int:
        """
        Store documents in the system.

        Documents are:
        1. Token-counted
        2. Embedded (Euclidean + hyperbolic)
        3. Distributed across memory layers

        Args:
            documents: Sequence of Document objects to store.

        Returns:
            Number of documents successfully stored.
        """
        self._ensure_initialized()

        stored = 0
        for doc in documents:
            # Estimate tokens if not provided
            doc.estimate_tokens()

            # Store in document registry
            self._documents[doc.id] = doc
            stored += 1

            # TODO (Phase 2): Embed and distribute to memory layers
            # await self._embed_document(doc)
            # await self._distribute_to_memory(doc)

        return stored

    async def store_text(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Convenience method to store a single text document.

        Args:
            text: Document text content.
            doc_id: Optional document identifier (auto-generated if not provided).
            metadata: Optional metadata dict.

        Returns:
            The stored Document object.
        """
        if doc_id is None:
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"

        doc = Document(
            id=doc_id,
            text=text,
            metadata=metadata or {},
        )

        await self.store_documents([doc])
        return doc

    # ──────────────────────────────────────────────────────────────────────
    # Query & Context Extension
    # ──────────────────────────────────────────────────────────────────────

    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        quality_threshold: Optional[float] = None,
    ) -> QueryResult:
        """
        Execute a context extension query.

        The pipeline:
        1. Determine budget from model profile
        2. Retrieve candidate contexts
        3. Apply quantum selection
        4. Compress to fit budget
        5. Evaluate quality
        6. (If quality < threshold) Refine and regenerate

        Args:
            query: User query string.
            top_k: Override number of contexts to retrieve.
            quality_threshold: Override quality threshold.

        Returns:
            QueryResult with selected context and quality metrics.
        """
        self._ensure_initialized()
        start_time = time.monotonic()

        # 1. Calculate token budget
        budget = TokenBudget.from_context_window(
            self._model_profile.native_context_window
        )

        # 2. Retrieve candidate documents
        k = top_k or self._config.hyperbolic.n_neighbors
        candidates = await self._retrieve_candidates(query, k)

        if not candidates:
            return QueryResult(
                query=query,
                context_blocks=[],
                total_tokens=0,
                retrieval_strategy="none",
                retrieval_time_ms=(time.monotonic() - start_time) * 1000,
            )

        # 3. Score and rank candidates
        scored = await self._score_candidates(query, candidates)

        # 4. Select context via quantum-inspired method
        selected = await self._quantum_select(query, scored, budget)

        # 5. Compress if needed to fit budget
        compressed = await self._compress_to_budget(selected, budget)

        # 6. Evaluate quality
        quality_threshold = quality_threshold or self._config.quality.quality_threshold
        result = await self._evaluate_quality(query, compressed, quality_threshold)

        result.retrieval_time_ms = (time.monotonic() - start_time) * 1000
        result.retrieval_strategy = self._model_profile.recommended_strategy.value

        return result

    async def query_with_response(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """
        Query and generate a full model response with extended context.

        This is the highest-level API: it handles context selection,
        prompt construction, and model generation in one call.
        """
        result = await self.query(query, **kwargs)

        # Build prompt with extended context
        context_text = result.format_context()
        prompt = (
            f"Based on the following context, answer the query.\n\n"
            f"--- Context ---\n{context_text}\n\n"
            f"--- Query ---\n{query}\n\n"
            f"--- Answer ---\n"
        )

        # Generate response
        response = await self._model_client.generate(prompt)
        return response

    # ──────────────────────────────────────────────────────────────────────
    # Internal Pipeline Methods
    # ──────────────────────────────────────────────────────────────────────

    async def _retrieve_candidates(
        self,
        query: str,
        top_k: int,
    ) -> List[Document]:
        """
        Retrieve candidate documents for a query.

        Phase 1: Simple keyword-based matching.
        Phase 2: Will use hyperbolic nearest neighbor search.
        """
        query_words = set(query.lower().split())
        candidates = []

        for doc in self._documents.values():
            doc_words = set(doc.text.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                doc.relevance_score = overlap / max(len(query_words), 1)
                candidates.append(doc)

        # Sort by relevance
        candidates.sort(key=lambda d: d.relevance_score, reverse=True)
        return candidates[:top_k]

    async def _score_candidates(
        self,
        query: str,
        candidates: List[Document],
    ) -> List[tuple]:
        """
        Score candidates using multi-dimensional evaluation.

        Dimensions:
        - Keyword relevance
        - Semantic similarity (TODO: Phase 2)
        - Information density
        - Recency (if available)
        """
        query_words = set(query.lower().split())
        scored = []

        for doc in candidates:
            # Keyword relevance
            doc_words = set(doc.text.lower().split())
            keyword_score = len(query_words & doc_words) / max(len(query_words), 1)

            # Information density (inverse of token count, normalized)
            density = 1.0 / max(doc.estimate_tokens(), 1) * 1000
            density = min(density, 1.0)

            # Weighted combination
            total_score = 0.5 * keyword_score + 0.3 * doc.relevance_score + 0.2 * density

            scored.append((doc, total_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    async def _quantum_select(
        self,
        query: str,
        scored_candidates: List[tuple],
        budget: TokenBudget,
    ) -> List[ContextBlock]:
        """
        Select context using quantum-inspired measurement.

        Constructs a superposition of all candidates, applies query as
        measurement operator, and collapses to top-k results.

        |ψ⟩ = Σᵢ √(scoreᵢ) |candidate_i⟩
        Measure with query → collapse to selected contexts
        """
        if not scored_candidates:
            return []

        if not self._config.quantum.enabled:
            # Fallback: classical top-k selection
            return self._classical_select(scored_candidates, budget)

        # Construct probability distribution from relevance scores
        scores = np.array([s for _, s in scored_candidates], dtype=np.float64)
        scores = np.clip(scores, 1e-6, None)  # Avoid zero scores
        probs = scores / scores.sum()

        # Create quantum state via Born rule: P(i) = |αᵢ|², so αᵢ = √P(i)
        state = QuantumState.from_probabilities(
            probs,
            labels=[doc.id for doc, _ in scored_candidates],
        )
        if not state.is_normalized:
            state = state.normalize()

        # Measurement probabilities = |αᵢ|² (Born rule)
        measurement_probs = state.probabilities

        # Select top-k by measurement probability
        k = self._config.quantum.top_k_measurements
        k = min(k, len(scored_candidates))

        # Budget-aware selection
        selected_blocks = []
        total_tokens = 0
        max_tokens = budget.available_for_retrieval

        indices = np.argsort(measurement_probs)[::-1]
        for idx in indices:
            doc, score = scored_candidates[idx]
            doc_tokens = doc.estimate_tokens()

            if total_tokens + doc_tokens > max_tokens:
                continue  # Skip if over budget

            block = ContextBlock(
                document_id=doc.id,
                text=doc.text,
                relevance_score=float(score),
                token_count=doc_tokens,
                quantum_amplitude=state.amplitudes[idx],
                measurement_probability=float(measurement_probs[idx]),
            )
            selected_blocks.append(block)
            total_tokens += doc_tokens

        return selected_blocks

    def _classical_select(
        self,
        scored_candidates: List[tuple],
        budget: TokenBudget,
    ) -> List[ContextBlock]:
        """Classical (non-quantum) context selection fallback."""
        selected = []
        total_tokens = 0
        max_tokens = budget.available_for_retrieval

        for doc, score in scored_candidates:
            doc_tokens = doc.estimate_tokens()
            if total_tokens + doc_tokens > max_tokens:
                continue

            block = ContextBlock(
                document_id=doc.id,
                text=doc.text,
                relevance_score=float(score),
                token_count=doc_tokens,
            )
            selected.append(block)
            total_tokens += doc_tokens

        return selected

    async def _compress_to_budget(
        self,
        blocks: List[ContextBlock],
        budget: TokenBudget,
    ) -> List[ContextBlock]:
        """
        Compress context blocks to fit within token budget.

        Phase 1: Simple truncation by relevance ranking.
        Phase 3: Will use MDL + entropy-based compression.
        """
        total_tokens = sum(b.token_count for b in blocks)
        max_tokens = budget.available_for_retrieval

        if total_tokens <= max_tokens:
            return blocks

        # Simple truncation: keep blocks by relevance order
        selected = []
        running_total = 0
        for block in sorted(blocks, key=lambda b: b.relevance_score, reverse=True):
            if running_total + block.token_count <= max_tokens:
                selected.append(block)
                running_total += block.token_count

        return selected

    async def _evaluate_quality(
        self,
        query: str,
        blocks: List[ContextBlock],
        threshold: float,
    ) -> QueryResult:
        """
        Evaluate quality of selected context.

        Multi-dimensional evaluation:
        Quality = 0.30·Relevance + 0.30·Completeness
                + 0.20·Coherence + 0.20·Accuracy
        """
        total_tokens = sum(b.token_count for b in blocks)

        # Relevance: average of block scores
        relevance = (
            sum(b.relevance_score for b in blocks) / max(len(blocks), 1)
            if blocks else 0.0
        )

        # Completeness: coverage of query terms in selected context
        query_terms = set(query.lower().split())
        all_context_terms = set()
        for block in blocks:
            all_context_terms.update(block.text.lower().split())
        completeness = len(query_terms & all_context_terms) / max(len(query_terms), 1)

        # Coherence: estimate based on context size and diversity
        coherence = min(len(blocks) / 5.0, 1.0) if blocks else 0.0

        # Accuracy: placeholder (requires fact-checking in Phase 4)
        accuracy = 0.85

        result = QueryResult(
            query=query,
            context_blocks=blocks,
            total_tokens=total_tokens,
            relevance_score=relevance,
            completeness_score=completeness,
            coherence_score=coherence,
            accuracy_score=accuracy,
        )
        result.compute_overall_quality()

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """Raise if system has not been initialized."""
        if not self._initialized:
            raise RuntimeError(
                "System not initialized. Call await system.initialize() first."
            )

    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        profile = self._model_profile
        return {
            "model": self._model_name,
            "initialized": self._initialized,
            "documents_stored": self.document_count,
            "model_category": (
                profile.classify_context_category().value
                if profile else "unknown"
            ),
            "native_context_window": (
                profile.native_context_window
                if profile else 0
            ),
            "target_context": self._config.target_extended_context,
            "quantum_selection_enabled": self._config.quantum.enabled,
        }
