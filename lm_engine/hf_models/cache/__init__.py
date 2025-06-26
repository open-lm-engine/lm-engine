# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from transformers import Cache

from ..config import CommonConfig
from .mamba2 import _Mamba2Cache
from .rnn import _RNNCache
from .softmax_attention import _SoftmaxAttentionCache


_CACHE_CLASSES = {
    "causal_convolution": _RNNCache,
    "gru": _RNNCache,
    "mamba2": _Mamba2Cache,
    "multihead_latent_attention": _SoftmaxAttentionCache,
    "rnn": _RNNCache,
    "softmax_attention": _SoftmaxAttentionCache,
    "stickbreaking_attention": _SoftmaxAttentionCache,
}


class GenerationCache(Cache):
    def __init__(self, config: CommonConfig, **kwargs) -> None:
        super().__init__()

        self._seen_tokens = 0
        self.cache: list[_SoftmaxAttentionCache] = [
            _CACHE_CLASSES[config.sequence_mixer_blocks[i].sequence_mixer_type](config, i, **kwargs)
            for i in range(config.num_layers)
        ]

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor]:
        return self.cache[layer_idx].get_cache()

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield self.cache[layer_idx].get_cache()

    def __len__(self):
        return self.seen_tokens

    def update(self, *, layer_idx: int, **kwargs) -> tuple[torch.Tensor | None]:
        return self.cache[layer_idx].update(**kwargs)

    def get_cache(self, layer_idx: int) -> torch.Tensor | tuple[torch.Tensor | None] | None:
        return self.cache[layer_idx].get_cache()

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return self.cache[layer_idx].get_seq_length()

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for cache in self.cache:
            cache.reorder_cache(beam_idx)
