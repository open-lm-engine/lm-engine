# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn

from .base import ParameterizedLinear


class LowRankLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_ranks: int,
        bias: bool = True,
        std_low_rank: float | None = None,
        std_high_rank: float | None = None,
    ) -> None:
        super().__init__()
        self.low_rank_proj = ParameterizedLinear(in_features, num_ranks, bias=bias, std=std_low_rank)
        self.high_rank_proj = ParameterizedLinear(num_ranks, out_features, bias=bias, std=std_high_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.low_rank_proj(x)
        x = self.high_rank_proj(x)
        return x
