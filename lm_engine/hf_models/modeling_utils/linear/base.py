# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....parameter import mark_parameter_as_initialized, mark_parameter_as_no_weight_decay
from ..quack import quack_linear


def linear_func(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    if is_kernel_allowed(Kernel.quack_gemm):
        # QuACK only handles local CUDA fp16/bf16 tensors; other Linear calls stay on torch.
        output = quack_linear(input, weight, bias)
        if output is not None:
            return output

    return F.linear(input, weight, bias)


class ParameterizedLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, std: float | None = None
    ) -> ParameterizedLinear:
        self.std = std
        super().__init__(in_features, out_features, bias)

        mark_parameter_as_no_weight_decay(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_func(x, self.weight, self.bias)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            nn.init.normal_(self.weight, mean=0, std=self.std)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias.zero_()

        mark_parameter_as_initialized(self.weight)
        mark_parameter_as_initialized(self.bias)
