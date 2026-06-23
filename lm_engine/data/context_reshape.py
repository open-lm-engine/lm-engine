# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations


class ContextReshapeDataset:
    """Wrap a dataset sampled at `source_sequence_length` (S) so it yields contiguous
    chunks of model context length `sequence_length` (L), where S is a multiple of L.

    Each base sample is a length-(S+1) token array; it is split into S/L overlapping-by-one
    windows of length (L+1), so the model sees identical tokens to an S-context run, only
    reset to context L. This lets a context-scaling sweep train every length on the exact
    same data.
    """

    def __init__(self, base_dataset, sequence_length: int, source_sequence_length: int) -> None:
        assert source_sequence_length % sequence_length == 0
        self.base_dataset = base_dataset
        self.sequence_length = sequence_length
        self.factor = source_sequence_length // sequence_length

    def __len__(self) -> int:
        return len(self.base_dataset) * self.factor

    def __getitem__(self, idx: int) -> dict:
        sample = self.base_dataset[idx // self.factor]["text"]
        assert len(sample) >= self.factor * self.sequence_length + 1, (
            f"base sample has {len(sample)} tokens; need >= {self.factor * self.sequence_length + 1} "
            "(source_sequence_length + 1)"
        )
        start = (idx % self.factor) * self.sequence_length
        return {"text": sample[start : start + self.sequence_length + 1]}
