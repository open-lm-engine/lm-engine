# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate

from ...dtensors import tensor_to_dtensor
from ..parameter import (
    mark_parameter_as_initialized,
    mark_parameter_as_mup_learning_rate,
    mark_parameter_as_no_weight_decay,
)
from .dtensor_module import DTensorModule
from .linear import ParameterizedLinear


class SoftplusDecayGate(DTensorModule):
    def __init__(
        self,
        hidden_size: int | None,
        output_size: int,
        std: float | None,
        has_projection: bool = False,
        A_init_min: float = 0,
        A_init_max: float = 16,
        dt_init_min: float = 1e-3,
        dt_init_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> SoftplusDecayGate:
        super().__init__()

        self.output_size = output_size
        self.has_projection = has_projection

        if has_projection:
            self.proj = ParameterizedLinear(hidden_size, self.output_size, std=std)
            mark_parameter_as_mup_learning_rate(self.proj.weight)
        else:
            assert hidden_size is None

        self.A_log = nn.Parameter(torch.empty(self.output_size, dtype=torch.float32))
        mark_parameter_as_no_weight_decay(self.A_log)

        self.dt_bias = nn.Parameter(torch.empty(self.output_size, dtype=torch.float32))
        mark_parameter_as_no_weight_decay(self.dt_bias)

        assert A_init_min >= 0
        assert A_init_max >= A_init_min

        self.A_init_min = A_init_min
        self.A_init_max = A_init_max

        assert dt_init_min > 0
        assert dt_init_max >= dt_init_min

        self.dt_init_min = dt_init_min
        self.dt_init_max = dt_init_max
        self.dt_init_floor = dt_init_floor

        self.reset_parameters()

    def forward(
        self, x: torch.Tensor, final_exponential: bool, output_dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        if self.has_projection:
            x = self.proj(x)

        x = x.float()
        x = x + self.dt_bias
        x = F.softplus(x)
        x = -self.A_log.float().exp() * x

        if final_exponential:
            x = torch.exp(x)

        x = x.to(output_dtype)

        return x

    @torch.no_grad()
    def reset_parameters(self) -> None:
        A = torch.empty(self.output_size, dtype=torch.float32).uniform_(self.A_init_min, self.A_init_max)

        if isinstance(self.A_log, DTensor):
            A = tensor_to_dtensor(
                tensor=A,
                device_mesh=self.A_log.device_mesh,
                current_placement=[Replicate()] * len(self.A_log.placements),
                desired_placement=self.A_log.placements,
            )

        self.A_log.copy_(torch.log(A))

        dt = torch.exp(
            torch.rand(self.output_size) * (math.log(self.dt_init_max) - math.log(self.dt_init_min))
            + math.log(self.dt_init_min)
        )
        dt = torch.clamp(dt, min=self.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        if isinstance(self.dt_bias, DTensor):
            inv_dt = tensor_to_dtensor(
                tensor=inv_dt,
                device_mesh=self.dt_bias.device_mesh,
                current_placement=[Replicate()] * len(self.dt_bias.placements),
                desired_placement=self.dt_bias.placements,
            )

        self.dt_bias.copy_(inv_dt)

        mark_parameter_as_initialized(self.A_log)
        mark_parameter_as_initialized(self.dt_bias)
