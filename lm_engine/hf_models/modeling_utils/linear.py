# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Replicate, Shard

from ...dtensors import dtensor_to_tensor, tensor_to_dtensor, use_async_tensor_parallel
from ...utils import ProcessGroupManager, divide_if_divisible
from ..parameter import mark_parameter_as_no_weight_decay
from .dtensor_module import DTensorModule
from .TP import get_module_placements


class ParameterizedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> ParameterizedLinear:
        self.std = std
        super().__init__(in_features, out_features, bias, device, dtype)

        mark_parameter_as_no_weight_decay(self.bias)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            nn.init.normal_(self.weight, mean=0, std=self.std)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias.zero_()


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
        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

        self.out_features_per_device = divide_if_divisible(
            out_features,
            self.tp_world_size,
            f"`out_features` ({out_features}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})",
        )

        super().__init__(
            in_features=in_features,
            out_features=self.out_features_per_device,
            bias=bias,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.weight = nn.Parameter(
            tensor_to_dtensor(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Shard(0)
            )
        )
        if bias:
            self.bias = nn.Parameter(
                tensor_to_dtensor(
                    self.bias, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Shard(0)
                )
            )

        self.input_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

        if use_async_tensor_parallel():
            self.compile()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(
            input, device_mesh=self.tp_mesh, current_placement=self.input_placement, desired_placement=Replicate()
        )
        input = super().forward(input)
        input = dtensor_to_tensor(input, device_mesh=self.tp_mesh, desired_placement=Shard(-1))
        return input

    def extra_repr(self) -> str:
        return "in_features={}, out_features_per_device={}, bias={}".format(
            self.in_features, self.out_features_per_device, self.bias is not None
        )


class RowParallelLinear(ParameterizedLinear, DTensorModule):
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
    ) -> RowParallelLinear:
        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

        self.in_features_per_device = divide_if_divisible(
            in_features,
            self.tp_world_size,
            f"`in_features` ({in_features}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})",
        )

        super().__init__(
            in_features=self.in_features_per_device,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            std=std,
        )

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

        self.output_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

        if use_async_tensor_parallel():
            self.compile()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.tp_world_size > 1:
            input = tensor_to_dtensor(input, device_mesh=self.tp_mesh, current_placement=Shard(-1))

        input = super().forward(input)

        if self.tp_world_size > 1:
            input = dtensor_to_tensor(input, device_mesh=self.tp_mesh, desired_placement=self.output_placement)

        return input

    def extra_repr(self) -> str:
        return "in_features_per_device={}, out_features={}, bias={}".format(
            self.in_features_per_device, self.out_features, self.bias is not None
        )
