# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from .linear import LinearCache


class ConstantCache(LinearCache):
    def __init__(self) -> ConstantCache:
        self.seen_tokens = 0
        self.cache: torch.Tensor | None = None

    def update(self, state: torch.Tensor | None = None, num_tokens_added: int = 0) -> torch.Tensor:
        self.seen_tokens += num_tokens_added

        if state is not None:
            self.cache = state

        return self.cache
