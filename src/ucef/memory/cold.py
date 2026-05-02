"""
Cold Memory Layer — File-System-based Long-Term Document Storage

The cold layer provides unlimited-capacity storage for the full document
archive. It uses HDF5 for efficient binary storage with optional compression,
or JSON for maximum portability.

Design:
- Priority: lowest (archival, historical)
- Latency target: < 500ms
- Storage: filesystem (HDF5, JSON, or Parquet)
- Capacity: unlimited
- Use case: full document archive, historical context
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ucef.core.config import ColdMemoryConfig
from ucef.core.types import Document


class FileSystemColdMemory:
    """
    Cold memory layer using filesystem storage.

    Supports three formats:
    - HDF5: efficient binary with compression (requires h5py)
    - JSON: maximum portability, no dependencies
    - Parquet: column-oriented, good for analytics (requires pyarrow)

    All formats store documents as:
    - id: str
    - text: str
    - metadata: JSON-serialized dict
    - token_count: int
    """

    def __init__(self, config: Optional[ColdMemoryConfig] = None) -> None:
        self._config = config or ColdMemoryConfig()
        self._storage_path = Path(self._config.storage_path)
        self._h5file: Any = None

        if self._config.enabled:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            if self._config.format == "hdf5":
                self._init_hdf5()

    def _init_hdf5(self) -> None:
        """Initialize HDF5 file handle."""
        try:
            import h5py
            h5_path = self._storage_path / "documents.h5"
            self._h5file = h5py.File(str(h5_path), "a")
        except ImportError:
            self._h5file = None
        except Exception:
            self._h5file = None

    # ──────────────────────────────────────────────────────────────────────
    # Core Operations
    # ──────────────────────────────────────────────────────────────────────

    async def store(self, doc: Document) -> bool:
        """Store a document in cold memory."""
        if not self._config.enabled:
            return False

        if self._config.format == "hdf5" and self._h5file is not None:
            return self._store_hdf5(doc)
        else:
            return self._store_json(doc)

    async def store_batch(self, docs: Sequence[Document]) -> int:
        """Store multiple documents."""
        stored = 0
        for doc in docs:
            if await self.store(doc):
                stored += 1
        return stored

    async def retrieve(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        if not self._config.enabled:
            return None

        if self._config.format == "hdf5" and self._h5file is not None:
            return self._retrieve_hdf5(doc_id)
        else:
            return self._retrieve_json(doc_id)

    async def retrieve_batch(self, doc_ids: Sequence[str]) -> List[Document]:
        """Retrieve multiple documents."""
        results = []
        for doc_id in doc_ids:
            doc = await self.retrieve(doc_id)
            if doc is not None:
                results.append(doc)
        return results

    async def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        if not self._config.enabled:
            return False

        if self._config.format == "hdf5" and self._h5file is not None:
            return self._delete_hdf5(doc_id)
        else:
            return self._delete_json(doc_id)

    async def list_ids(self) -> List[str]:
        """List all stored document IDs."""
        if not self._config.enabled:
            return []

        if self._config.format == "hdf5" and self._h5file is not None:
            return list(self._h5file.keys())
        else:
            return [
                p.stem for p in self._storage_path.glob("*.json")
            ]

    async def count(self) -> int:
        """Count stored documents."""
        ids = await self.list_ids()
        return len(ids)

    # ──────────────────────────────────────────────────────────────────────
    # JSON Implementation
    # ──────────────────────────────────────────────────────────────────────

    def _store_json(self, doc: Document) -> bool:
        """Store document as JSON file."""
        try:
            path = self._storage_path / f"{doc.id}.json"
            data = {
                "id": doc.id,
                "text": doc.text,
                "metadata": doc.metadata,
                "token_count": doc.estimate_tokens(),
            }
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except Exception:
            return False

    def _retrieve_json(self, doc_id: str) -> Optional[Document]:
        """Retrieve document from JSON file."""
        try:
            path = self._storage_path / f"{doc_id}.json"
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return Document(
                id=data["id"],
                text=data["text"],
                metadata=data.get("metadata", {}),
                token_count=data.get("token_count", 0),
            )
        except Exception:
            return None

    def _delete_json(self, doc_id: str) -> bool:
        """Delete JSON file."""
        try:
            path = self._storage_path / f"{doc_id}.json"
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────────
    # HDF5 Implementation
    # ──────────────────────────────────────────────────────────────────────

    def _store_hdf5(self, doc: Document) -> bool:
        """Store document in HDF5 file."""
        try:
            compression = self._config.compression if self._config.compression != "none" else None

            if doc.id in self._h5file:
                del self._h5file[doc.id]

            grp = self._h5file.create_group(doc_id)
            grp.attrs["text"] = doc.text
            grp.attrs["metadata"] = json.dumps(doc.metadata)
            grp.attrs["token_count"] = doc.estimate_tokens()

            self._h5file.flush()
            return True
        except Exception:
            return False

    def _retrieve_hdf5(self, doc_id: str) -> Optional[Document]:
        """Retrieve document from HDF5 file."""
        try:
            if doc_id not in self._h5file:
                return None
            grp = self._h5file[doc_id]
            return Document(
                id=doc_id,
                text=grp.attrs["text"],
                metadata=json.loads(grp.attrs.get("metadata", "{}")),
                token_count=int(grp.attrs.get("token_count", 0)),
            )
        except Exception:
            return None

    def _delete_hdf5(self, doc_id: str) -> bool:
        """Delete from HDF5 file."""
        try:
            if doc_id in self._h5file:
                del self._h5file[doc_id]
                self._h5file.flush()
                return True
            return False
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close any open file handles."""
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:
                pass
            self._h5file = None

    def __del__(self) -> None:
        self.close()
