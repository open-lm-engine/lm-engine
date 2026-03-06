# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Replicate, Shard

from ....dtensors import dtensor_to_tensor, tensor_to_dtensor, use_async_tensor_parallel
from ....utils import ProcessGroupManager, divide_if_divisible
from ..dtensor_module import DTensorModule
from ..TP import get_module_placements
from .base import ParameterizedLinear


class RowParallelLinear(ParameterizedLinear, DTensorModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        std: float | None = None,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> RowParallelLinear:
        DTensorModule.__init__(self)

        self.in_features_per_tp_rank = divide_if_divisible(
            in_features,
            self.tp_world_size,
            f"`in_features` ({in_features}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})",
        )

        ParameterizedLinear.__init__(
            self, in_features=self.in_features_per_tp_rank, out_features=out_features, bias=bias, std=std
        )

        if self.is_tp_enabled:
            self.output_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

            self.weight = nn.Parameter(
                tensor_to_dtensor(
                    self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Shard(1)
                )
            )

            if bias:
                self.bias = nn.Parameter(
                    tensor_to_dtensor(
                        self.bias,
                        device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                        current_placement=Replicate(),
                    )
                )

            if use_async_tensor_parallel():
                self.compile()

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_tp_enabled:
            x = tensor_to_dtensor(x, device_mesh=self.tp_mesh, current_placement=Shard(-1))
            x = super().forward(x)
            x = dtensor_to_tensor(x, device_mesh=self.tp_mesh, desired_placement=self.output_placement)
        else:
            x = super().forward(x)

        return x

    def extra_repr(self) -> str:
        return "in_features_per_tp_rank={}, out_features={}, bias={}".format(
            self.in_features_per_tp_rank, self.out_features, self.bias is not None
        )
