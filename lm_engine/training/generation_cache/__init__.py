# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .constant import ConstantCache
from .linear import LinearCache


CACHE_TYPE = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
LAYER_CACHE_TYPE = tuple[ConstantCache | LinearCache, ...]


@dataclass
class GenerationState:
    state: torch.Tensor
    method: ConstantCache | LinearCache
    num_tokens_added: int | None = None


class GenerationCache:
    def __init__(self) -> GenerationCache:
        self.cache: list[LAYER_CACHE_TYPE] = []
        self.named_cache: dict[str, dict[int, LAYER_CACHE_TYPE]] = {}

    def __getitem__(self, layer_idx: int) -> CACHE_TYPE:
        return tuple(cache.get_cache() for cache in self.cache[layer_idx])

    def __iter__(self) -> Iterable[CACHE_TYPE]:
        for layer_idx in range(len(self)):
            yield tuple(cache.get_cache() for cache in self.cache[layer_idx])

    def update(
        self, states: tuple[GenerationState], layer_idx: int, cache_name: str | None = None
    ) -> list[torch.Tensor]:
        assert isinstance(states, tuple)

        if cache_name is None:
            if len(self.cache) == layer_idx:
                self.cache.append(tuple(state.method() for state in states))

            layer_cache = self.cache[layer_idx]
        else:
            namespace = self.named_cache.setdefault(cache_name, {})
            if layer_idx not in namespace:
                namespace[layer_idx] = tuple(state.method() for state in states)
            layer_cache = namespace[layer_idx]

        assert len(states) == len(layer_cache)

        output_state = []
        for state, cache in zip(states, layer_cache):
            assert type(cache) == state.method

            kwargs = {"state": state.state}
            if state.num_tokens_added is not None:
                kwargs["num_tokens_added"] = state.num_tokens_added

            output_state.append(cache.update(**kwargs))

        return output_state

    def get_cache(self, layer_idx: int, empty_value: tuple[None] | None, cache_name: str | None = None) -> CACHE_TYPE:
        if cache_name is None:
            if len(self.cache) == layer_idx:
                return empty_value

            layer_cache = self.cache[layer_idx]
        else:
            layer_cache = self.named_cache.get(cache_name, {}).get(layer_idx)
            if layer_cache is None:
                return empty_value

        return tuple(cache.get_cache() for cache in layer_cache)

    def get_seq_length(self, layer_idx: int = 0, cache_name: str | None = None) -> int:
        if cache_name is None:
            if len(self.cache) == layer_idx:
                return 0

            layer_cache = self.cache[layer_idx]
        else:
            layer_cache = self.named_cache.get(cache_name, {}).get(layer_idx)
            if layer_cache is None:
                return 0

        lengths = [cache.get_seq_length() for cache in layer_cache]
        match = [i == lengths[0] for i in lengths]
        assert all(match)

        return lengths[0]

    def _iter_layer_caches(self) -> Iterable[LAYER_CACHE_TYPE]:
        yield from self.cache
        for namespace in self.named_cache.values():
            yield from namespace.values()

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for layer_cache in self._iter_layer_caches():
            for cache in layer_cache:
                cache.reorder_cache(beam_idx)
