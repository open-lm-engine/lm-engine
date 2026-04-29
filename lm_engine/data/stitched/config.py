# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class OrderingStrategy(Enum):
    """How to order documents within a collection at fetch time."""

    as_stored = "as_stored"  # use the default ordering of the given processed/tokenized data
    reversed = "reversed"  # reverse the ordering of the given processed/tokenized data
    shuffled = "shuffled"  # shuffle the ordering of the given processed/tokenized data


@dataclass
class StitchedDatasetConfig:
    """Configuration for StitchedSequenceDataset.

    Args:
        stitched_seq_path: Path to the parquet file with columns [shard, coll_beg, coll_end].
        tokenized_data_root: Root directory that contains shard subdirectories.
        sequence_length: Number of tokens per training sample (not counting the extra label token).
        ordering_strategy: How to order documents within a collection at fetch time.
        seed: Random seed used for shuffled ordering.
        split_ratio: (train, val, test) fractions that sum to 1.0.
            E.g. (0.98, 0.01, 0.01).  Applied over the *collections* dimension.
        cache_dir: Auto-computed from stitched_seq_path and hyperparams (read-only property).
    """

    stitched_seq_path: str
    tokenized_data_root: str
    sequence_length: int
    ordering_strategy: OrderingStrategy = OrderingStrategy.as_stored
    seed: int = 42
    split_ratio: tuple[float, float, float] = field(default_factory=lambda: (1.0, 0.0, 0.0))

    @property
    def cache_dir(self) -> Path:
        split_str = "-".join(str(r) for r in self.split_ratio)
        order_str = self.ordering_strategy.value
        if self.ordering_strategy == OrderingStrategy.shuffled:
            order_str += f"_seed{self.seed}"
        name = f"cache_seqlen{self.sequence_length}_split{split_str}_{order_str}"
        return Path(self.stitched_seq_path).parent / name
