# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch


class _LinearCache:
    def __init__(self) -> _LinearCache:
        self.seen_tokens = 0
        self.cache: torch.Tensor | None = None

    def get_cache(self) -> tuple[torch.Tensor | None]:
        return self.cache

    def update(self, state: torch.Tensor, sequence_length_dimension: int = 1) -> torch.Tensor:
        self.seen_tokens += state.size(sequence_length_dimension)

        if self.cache is None:
            self.cache = state
        else:
            self.cache = torch.cat([self.cache, state], dim=sequence_length_dimension)

        return self.cache

    def get_seq_length(self) -> int:
        return self.seen_tokens

    def get_max_cache_shape(self) -> None:
        return None

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        self.cache = self.cache.index_select(0, beam_idx.to(self.cache.device))
