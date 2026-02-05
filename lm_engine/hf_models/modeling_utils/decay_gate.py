# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..parameter import (
    mark_parameter_as_initialized,
    mark_parameter_as_mup_learning_rate,
    mark_parameter_as_no_weight_decay,
)
from .linear import ParameterizedLinear


class SoftplusDecayGate(nn.Module):
    def __init__(
        self, hidden_size: int, output_size: int, std: float | None, has_projection: bool = False
    ) -> SoftplusDecayGate:
        super().__init__()

        self.output_size = output_size
        self.has_projection = has_projection

        if has_projection:
            self.proj = ParameterizedLinear(hidden_size, self.output_size, std=std)
            mark_parameter_as_mup_learning_rate(self.proj.weight)

        self.A_log = nn.Parameter(torch.empty(self.output_size, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.output_size, dtype=torch.float32))

        self.reset_parameters()
        mark_parameter_as_no_weight_decay(self.dt_bias)

    def forward(self, x: torch.Tensor, final_exponential: bool) -> torch.Tensor:
        dtype = x.dtype

        if self.has_projection:
            x = self.proj(x)

        x = x.float()
        x = x + self.dt_bias
        x = F.softplus(x)
        x = -self.A_log.float().exp() * x

        if final_exponential:
            x = torch.exp(x)

        x = x.to(dtype)

        return x

    @torch.no_grad()
    def reset_parameters(self) -> None:
        A = torch.arange(1, self.output_size + 1, dtype=torch.float32)
        self.A_log.copy_(torch.log(A))

        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(torch.rand(self.output_size) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.copy_(inv_dt)

        mark_parameter_as_initialized(self.A_log)
        mark_parameter_as_initialized(self.dt_bias)
