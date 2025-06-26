# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ..config import CommonConfig
from .softmax_attention import _SoftmaxAttentionCache


class _RNNCache(_SoftmaxAttentionCache):
    def __init__(self, config: CommonConfig, layer_idx: int, **kwargs) -> None:
        self.seen_tokens = 0
        self.cache: torch.Tensor | None = None

    def get_cache(self) -> torch.Tensor | None:
        return self.cache

    def update(self, state: torch.Tensor | None = None, num_tokens_added: int = 0) -> torch.Tensor:
        self.seen_tokens += num_tokens_added

        if state is not None:
            self.cache = state

        return self.cache

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        self.cache = self.cache.index_select(0, beam_idx.to(self.cache.device))
