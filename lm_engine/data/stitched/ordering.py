# **************************************************
# Copyright (c) 2026, Mayank Mishra, Jyo Pari, Zhonglin Han
# **************************************************

from __future__ import annotations

import numpy as np

from .config import OrderingStrategy


def get_doc_order(
    beg: int,
    end: int,
    strategy: OrderingStrategy,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Return the document indices [beg, end) in the desired order.

    Args:
        beg: First document index in the collection (inclusive).
        end: Last document index in the collection (exclusive).
        strategy: Ordering strategy to apply.
        rng: NumPy random state; required for OrderingStrategy.shuffled.

    Returns:
        1-D int64 array of length (end - beg).
    """
    indices = np.arange(beg, end, dtype=np.int64)

    if strategy == OrderingStrategy.as_stored:
        return indices

    if strategy == OrderingStrategy.reversed:
        return indices[::-1]

    if strategy == OrderingStrategy.shuffled:
        assert rng is not None, "rng must be provided for shuffled ordering"
        rng.shuffle(indices)
        return indices

    raise ValueError(f"Unknown ordering strategy: {strategy}")
