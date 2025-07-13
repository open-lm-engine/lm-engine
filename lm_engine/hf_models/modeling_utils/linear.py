# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..parameter import mark_parameter_as_no_weight_decay
from .normalization import get_normalization_function


class ParameterizedLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, std: float | None = None
    ) -> ParameterizedLinear:
        self.std = std
        super().__init__(in_features, out_features, bias)

        mark_parameter_as_no_weight_decay(self.bias)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            nn.init.normal_(self.weight, mean=0, std=self.std)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias.zero_()


class ParameterizedLowRankLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        norm: bool = False,
        std: float | None = None,
    ) -> ParameterizedLowRankLinear:
        super().__init__()

        if not norm:
            std = math.sqrt(std / math.sqrt(rank))

        self.l1 = ParameterizedLinear(in_features, rank, bias=bias, std=std)
        self.norm = get_normalization_function("rmsnorm" if norm else None, rank)
        self.l2 = ParameterizedLinear(rank, out_features, bias=bias, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.norm(x)
        x = self.l2(x)
        return x
