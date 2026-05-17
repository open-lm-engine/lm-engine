# **************************************************
# Copyright (c) 2026, Jyo Pari, Mayank Mishra
# **************************************************

from __future__ import annotations

import os

import numpy as np

from ..megatron.indexed_dataset import MMapIndexedDataset


class ShardStore:
    """Lazy-loading store of MMapIndexedDataset shards with LRU eviction.

    Workers in different DataLoader processes each hold their own ShardStore.
    The store never crosses process boundaries (MMapIndexedDataset is
    re-opened in each worker after spawn via __getstate__/__setstate__).
    Each worker has its own ShardStore instance.

    Args:
        tokenized_data_root: Root directory containing shard subdirectories.
        max_open: Maximum number of shards to keep open simultaneously.
    """

    def __init__(self, tokenized_data_root: str, max_open: int = 64) -> None:
        self.tokenized_data_root = tokenized_data_root
        self.max_open = max_open
        self._cache: dict[str, MMapIndexedDataset] = {}
        self._order: list[str] = []  # LRU order; most-recently-used at the back

    def get_tokens(self, shard_id: str, doc_idx: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        """Return a token array for *doc_idx* inside *shard_id*.

        Args:
            shard_id: Relative shard path (e.g. "cc/shard_0001_content").
            doc_idx: Absolute document index within the shard.
            offset: Token offset within the document (for partial reads).
            length: Number of tokens to read; None means read to end of doc.
        """
        ds = self._get_dataset(shard_id)
        return ds.get(doc_idx, offset=offset, length=length)

    def get_sequence_lengths(self, shard_id: str) -> np.ndarray:
        """Return the full sequence_lengths array for a shard (read-only view)."""
        return self._get_dataset(shard_id).sequence_lengths

    def _get_dataset(self, shard_id: str) -> MMapIndexedDataset:
        if shard_id in self._cache:
            # Promote to most-recently-used
            self._order.remove(shard_id)
            self._order.append(shard_id)
            return self._cache[shard_id]

        # Evict the least-recently-used shard if at capacity
        if len(self._cache) >= self.max_open:
            lru = self._order.pop(0)
            del self._cache[lru]

        path_prefix = os.path.join(self.tokenized_data_root, shard_id)
        ds = MMapIndexedDataset(path_prefix)
        self._cache[shard_id] = ds
        self._order.append(shard_id)
        return ds

    def __getstate__(self) -> dict:
        # Serialize only config
        return {"tokenized_data_root": self.tokenized_data_root, "max_open": self.max_open}

    def __setstate__(self, state: dict) -> None:
        # Reconstruct in the worker with empty cache
        self.tokenized_data_root = state["tokenized_data_root"]
        self.max_open = state["max_open"]
        self._cache = {}
        self._order = []
