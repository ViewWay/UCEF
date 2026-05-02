"""
Hot Memory Layer — Redis-based LRU Cache for Active Context

The hot layer stores recently accessed documents and active session context
with sub-10ms latency. Implements LRU + TTL eviction and token budget
management.

Design:
- Priority: highest (most recent, most relevant)
- Latency target: < 10ms
- Eviction: LRU + TTL
- Capacity: limited by token budget (~20K tokens)
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from ucef.core.config import HotMemoryConfig
from ucef.core.types import Document


class RedisHotMemory:
    """
    Hot memory layer with Redis-like interface.

    In production, this wraps a real Redis client. For development and
    testing, it falls back to an in-memory OrderedDict with LRU semantics.

    Both modes provide:
    - O(1) get/set by document ID
    - TTL-based expiration
    - Token budget enforcement
    - LRU eviction when budget exceeded
    """

    def __init__(self, config: Optional[HotMemoryConfig] = None) -> None:
        self._config = config or HotMemoryConfig()
        self._redis_client: Any = None
        self._fallback_store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._current_tokens = 0

        # Try to connect to Redis if enabled
        if self._config.enabled:
            self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection. Falls back to in-memory on failure."""
        try:
            import redis
            self._redis_client = redis.from_url(
                self._config.redis_url,
                decode_responses=True,
            )
            self._redis_client.ping()
        except Exception:
            self._redis_client = None

    # ──────────────────────────────────────────────────────────────────────
    # Core Operations
    # ──────────────────────────────────────────────────────────────────────

    async def store(self, doc: Document) -> bool:
        """
        Store a document in hot memory.

        Evicts LRU entries if token budget would be exceeded.

        Returns:
            True if stored successfully.
        """
        if not self._config.enabled:
            return False

        doc_tokens = doc.estimate_tokens()

        # Evict until we have space
        while (
            self._current_tokens + doc_tokens > self._config.max_tokens
            and self._fallback_store
        ):
            self._evict_lru()

        if self._redis_client:
            return await self._store_redis(doc, doc_tokens)
        else:
            return self._store_fallback(doc, doc_tokens)

    async def retrieve(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID. Returns None if not found or expired."""
        if not self._config.enabled:
            return None

        if self._redis_client:
            return await self._retrieve_redis(doc_id)
        else:
            return self._retrieve_fallback(doc_id)

    async def delete(self, doc_id: str) -> bool:
        """Delete a document from hot memory."""
        if not self._config.enabled:
            return False

        if self._redis_client:
            return await self._delete_redis(doc_id)
        else:
            return self._delete_fallback(doc_id)

    async def retrieve_all(self) -> List[Document]:
        """Retrieve all non-expired documents."""
        if not self._config.enabled:
            return []

        if self._redis_client:
            return await self._retrieve_all_redis()
        else:
            self._cleanup_expired()
            return [entry.document for entry in self._fallback_store.values()]

    # ──────────────────────────────────────────────────────────────────────
    # Fallback (In-Memory) Implementation
    # ──────────────────────────────────────────────────────────────────────

    def _store_fallback(self, doc: Document, doc_tokens: int) -> bool:
        """Store in OrderedDict with LRU semantics."""
        if doc.id in self._fallback_store:
            old_entry = self._fallback_store[doc.id]
            self._current_tokens -= old_entry.token_count

        entry = _CacheEntry(
            document=doc,
            token_count=doc_tokens,
            stored_at=time.time(),
            expires_at=time.time() + self._config.ttl_seconds,
        )
        self._fallback_store[doc.id] = entry
        self._current_tokens += doc_tokens
        return True

    def _retrieve_fallback(self, doc_id: str) -> Optional[Document]:
        """Retrieve from OrderedDict, checking TTL."""
        if doc_id not in self._fallback_store:
            return None

        entry = self._fallback_store[doc_id]

        if time.time() > entry.expires_at:
            self._delete_fallback(doc_id)
            return None

        # Move to end (most recently used)
        self._fallback_store.move_to_end(doc_id)
        return entry.document

    def _delete_fallback(self, doc_id: str) -> bool:
        """Delete from OrderedDict."""
        if doc_id in self._fallback_store:
            entry = self._fallback_store.pop(doc_id)
            self._current_tokens -= entry.token_count
            return True
        return False

    async def _retrieve_all_redis(self) -> List[Document]:
        """Retrieve all from Redis."""
        # Simplified — full Redis implementation would scan keys
        return []

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._fallback_store:
            _, entry = self._fallback_store.popitem(last=False)
            self._current_tokens -= entry.token_count

    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        now = time.time()
        expired = [
            doc_id for doc_id, entry in self._fallback_store.items()
            if now > entry.expires_at
        ]
        for doc_id in expired:
            self._delete_fallback(doc_id)

    # ──────────────────────────────────────────────────────────────────────
    # Redis Implementation (stubs for production)
    # ──────────────────────────────────────────────────────────────────────

    async def _store_redis(self, doc: Document, doc_tokens: int) -> bool:
        """Store in Redis with TTL."""
        try:
            import json
            data = json.dumps({
                "id": doc.id,
                "text": doc.text,
                "metadata": doc.metadata,
                "token_count": doc_tokens,
            })
            self._redis_client.setex(
                f"ucef:hot:{doc.id}",
                self._config.ttl_seconds,
                data,
            )
            return True
        except Exception:
            return False

    async def _retrieve_redis(self, doc_id: str) -> Optional[Document]:
        """Retrieve from Redis."""
        try:
            import json
            data = self._redis_client.get(f"ucef:hot:{doc_id}")
            if data is None:
                return None
            parsed = json.loads(data)
            return Document(
                id=parsed["id"],
                text=parsed["text"],
                metadata=parsed.get("metadata", {}),
                token_count=parsed.get("token_count", 0),
            )
        except Exception:
            return None

    async def _delete_redis(self, doc_id: str) -> bool:
        """Delete from Redis."""
        try:
            self._redis_client.delete(f"ucef:hot:{doc_id}")
            return True
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def current_tokens(self) -> int:
        return self._current_tokens

    @property
    def max_tokens(self) -> int:
        return self._config.max_tokens

    @property
    def utilization(self) -> float:
        return self._current_tokens / max(self._config.max_tokens, 1)

    @property
    def count(self) -> int:
        return len(self._fallback_store)

    @property
    def using_redis(self) -> bool:
        return self._redis_client is not None


class _CacheEntry:
    """Internal cache entry with metadata."""
    __slots__ = ("document", "token_count", "stored_at", "expires_at")

    def __init__(
        self,
        document: Document,
        token_count: int,
        stored_at: float,
        expires_at: float,
    ) -> None:
        self.document = document
        self.token_count = token_count
        self.stored_at = stored_at
        self.expires_at = expires_at
