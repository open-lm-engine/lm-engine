# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from .rnn import _RNNCache


class _CausalConvolutionCache(_RNNCache):
    def update(self, state: torch.Tensor | None = None, num_tokens_added: int = 0) -> torch.Tensor:
        self.seen_tokens += num_tokens_added

        if state is not None:
            self.cache = state

        return self.cache
