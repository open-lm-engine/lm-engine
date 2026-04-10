# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from .constant import ConstantCache
from .linear import LinearCache


CACHE_TYPE = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None


@dataclass
class GenerationState:
    state: torch.Tensor
    method: ConstantCache | LinearCache
    kwargs: dict = field(default_factory=dict)


class GenerationCache:
    def __init__(self) -> GenerationCache:
        self.cache: list[tuple[ConstantCache | LinearCache]] = []

    def __getitem__(self, layer_idx: int) -> CACHE_TYPE:
        return self.cache[layer_idx].get_cache()

    def __iter__(self) -> Iterable[CACHE_TYPE]:
        for layer_idx in range(len(self)):
            yield self.cache[layer_idx].get_cache()

    def update(self, states: tuple[GenerationState], layer_idx: int) -> CACHE_TYPE:
        assert isinstance(states, tuple)
        output_state = []

        if len(self.cache) == layer_idx:
            layer_cache = []
            for state in states:
                layer_cache.append(state.method())
                output_state.append(layer_cache[-1].update(state=state.state, **state.kwargs))

            self.cache.append(tuple(layer_cache))
        else:
            layer_cache = self.cache[layer_idx]
            for state, cache in zip(states, layer_cache):
                assert type(cache) == state.method
                output_state.append(cache.update(state=state.state, **state.kwargs))

        return output_state

    def get_cache(self, layer_idx: int) -> CACHE_TYPE:
        return tuple(cache.get_cache() for cache in self.cache[layer_idx])

    def get_seq_length(self, layer_idx: int = 0) -> int:

        return self.cache[layer_idx].get_seq_length()

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for layer_cache in self.cache:
            for cache in layer_cache:
                cache.reorder_cache(beam_idx)
