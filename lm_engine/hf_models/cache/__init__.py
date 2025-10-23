# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Iterable

import torch

from ..config import CommonConfig
from .attention import _SoftmaxAttentionCache
from .mamba2 import _Mamba2Cache
from .rnn import _RNNCache


_CACHE_CLASSES = {
    "causal_convolution": _RNNCache,
    "gru": _RNNCache,
    "mamba2": _Mamba2Cache,
    "multihead_latent_attention": _SoftmaxAttentionCache,
    "rnn": _RNNCache,
    "rsa": _Mamba2Cache,
    "softmax_attention": _SoftmaxAttentionCache,
}

CACHE_TYPE = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None


class GenerationCache:
    def __init__(self, config: CommonConfig, **kwargs) -> GenerationCache:
        self.cache: list[_SoftmaxAttentionCache] = [
            _CACHE_CLASSES[config.sequence_mixer_blocks[i].sequence_mixer_type](config, i, **kwargs)
            for i in range(config.num_layers)
        ]

    def __getitem__(self, layer_idx: int) -> CACHE_TYPE:
        return self.cache[layer_idx].get_cache()

    def __iter__(self) -> Iterable[CACHE_TYPE]:
        for layer_idx in range(len(self)):
            yield self.cache[layer_idx].get_cache()

    def update(self, *, layer_idx: int, **kwargs) -> CACHE_TYPE:
        return self.cache[layer_idx].update(**kwargs)

    # TODO remove this function
    def get_cache(self, layer_idx: int) -> CACHE_TYPE:
        return self.cache[layer_idx].get_cache()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.cache[layer_idx].get_seq_length()

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for cache in self.cache:
            cache.reorder_cache(beam_idx)
