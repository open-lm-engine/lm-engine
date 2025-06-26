# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ..config import CommonConfig
from .rnn import _RNNCache
from .softmax_attention import _SoftmaxAttentionCache


class _Mamba2Cache(_SoftmaxAttentionCache):
    def __init__(self, config: CommonConfig, layer_idx: int, **kwargs) -> None:
        self.seen_tokens = 0
        self.conv_cache = _RNNCache(config, layer_idx, **kwargs)
        self.ssm_cache = _RNNCache(config, layer_idx, **kwargs)

    def get_cache(self) -> tuple[torch.Tensor | None]:
        return self.conv_cache.get_cache(), self.ssm_cache.get_cache()

    def update(
        self, conv_state: torch.Tensor | None = None, ssm_state: torch.Tensor | None = None, num_tokens_added: int = 0
    ) -> tuple[torch.Tensor]:
        self.seen_tokens += num_tokens_added
        conv_cache = self.conv_cache.update(conv_state, num_tokens_added=num_tokens_added)
        ssm_cache = self.ssm_cache.update(ssm_state, num_tokens_added=num_tokens_added)
        return conv_cache, ssm_cache

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        self.conv_cache.reorder_cache(beam_idx)
        self.ssm_cache.reorder_cache(beam_idx)
