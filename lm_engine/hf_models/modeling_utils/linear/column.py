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


class ColumnParallelLinear(ParameterizedLinear, DTensorModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> ColumnParallelLinear:
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        self.out_features_per_tp_rank = divide_if_divisible(
            out_features,
            tp_world_size,
            f"`out_features` ({out_features}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
        )

        super().__init__(
            in_features=in_features,
            out_features=self.out_features_per_tp_rank,
            bias=bias,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.is_tp_enabled = ProcessGroupManager.is_tensor_parallel_enabled()

        if self.is_tp_enabled:
            self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()
            self.input_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

            self.weight = nn.Parameter(
                tensor_to_dtensor(
                    self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Shard(0)
                )
            )

            if bias:
                self.bias = nn.Parameter(
                    tensor_to_dtensor(
                        self.bias,
                        device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                        current_placement=Shard(0),
                    )
                )

            if use_async_tensor_parallel():
                self.compile()

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_tp_enabled:
            x = tensor_to_dtensor(
                x, device_mesh=self.tp_mesh, current_placement=self.input_placement, desired_placement=Replicate()
            )

        x = super().forward(x)

        if self.is_tp_enabled:
            x = dtensor_to_tensor(x, device_mesh=self.tp_mesh, desired_placement=Shard(-1))

        return x

    def extra_repr(self) -> str:
        return "in_features={}, out_features_per_tp_rank={}, bias={}".format(
            self.in_features, self.out_features_per_tp_rank, self.bias is not None
        )
