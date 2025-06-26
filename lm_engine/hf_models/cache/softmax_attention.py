# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ..config import CommonConfig


class _SoftmaxAttentionCache:
    def __init__(self, config: CommonConfig, layer_idx: int, **kwargs) -> None:
        self.seen_tokens = 0
        self.key_cache: torch.Tensor | None = None
        self.value_cache: torch.Tensor | None = None

    def get_cache(self) -> tuple[torch.Tensor | None]:
        return self.key_cache, self.value_cache

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor | None = None, sequence_length_dimension: int = -2
    ) -> tuple[torch.Tensor]:
        self.seen_tokens += key_states.size(sequence_length_dimension)

        if self.key_cache is None:
            self.key_cache = key_states
        else:
            self.key_cache = torch.cat([self.key_cache, key_states], dim=sequence_length_dimension)

        if value_states is not None:
            if self.value_cache is None:
                self.value_cache = value_states
            else:
                self.value_cache = torch.cat([self.value_cache, value_states], dim=sequence_length_dimension)

        return self.key_cache, self.value_cache

    def get_seq_length(self) -> int:
        return self.seen_tokens

    def get_max_cache_shape(self) -> None:
        return None

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        self.key_cache = self.key_cache.index_select(0, beam_idx.to(self.key_cache.device))
        self.value_cache = self.value_cache.index_select(0, beam_idx.to(self.value_cache.device))
