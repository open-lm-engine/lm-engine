# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .constant import ConstantCache
from .linear import LinearCache


CACHE_TYPE = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None


@dataclass
class GenerationState:
    state: torch.Tensor
    method: ConstantCache | LinearCache
    num_tokens_added: int | None = None


class GenerationCache:
    def __init__(self) -> GenerationCache:
        self.cache: list[tuple[ConstantCache | LinearCache]] = []

    def __getitem__(self, layer_idx: int) -> CACHE_TYPE:
        return tuple(cache.get_cache() for cache in self.cache[layer_idx])

    def __iter__(self) -> Iterable[CACHE_TYPE]:
        for layer_idx in range(len(self)):
            yield tuple(cache.get_cache() for cache in self.cache[layer_idx])

    def update(self, states: tuple[GenerationState], layer_idx: int) -> CACHE_TYPE:
        assert isinstance(states, tuple)

        if len(self.cache) == layer_idx:
            self.cache.append(tuple(state.method() for state in states))

        layer_cache = self.cache[layer_idx]
        assert len(states) == len(layer_cache)

        output_state = []
        for state, cache in zip(states, layer_cache):
            assert type(cache) == state.method

            kwargs = {"state": state.state}
            if state.num_tokens_added is not None:
                kwargs["num_tokens_added"] = state.num_tokens_added

            output_state.append(cache.update(**kwargs))

        return output_state

    def get_cache(self, layer_idx: int, empty_value: tuple[None] | None) -> CACHE_TYPE:
        if len(self.cache) == layer_idx:
            return empty_value

        return tuple(cache.get_cache() for cache in self.cache[layer_idx])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.cache) == layer_idx:
            return 0

        lengths = [cache.get_seq_length() for cache in self.cache[layer_idx]]
        match = [i == lengths[0] for i in lengths]
        assert all(match)

        return lengths[0]

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for layer_cache in self.cache:
            for cache in layer_cache:
                cache.reorder_cache(beam_idx)
