# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch.nn as nn
from torch.distributed._tensor.placement_types import Replicate

from ....dtensors import tensor_to_dtensor
from ....utils import ProcessGroupManager
from ..dtensor_module import DTensorModule
from .base import ParameterizedLinear


class ReplicatedLinear(ParameterizedLinear, DTensorModule):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, std: float | None = None
    ) -> ReplicatedLinear:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, std=std)

        self.is_tp_enabled = ProcessGroupManager.is_tensor_parallel_enabled()

        if self.is_tp_enabled:
            self.weight = nn.Parameter(
                tensor_to_dtensor(
                    self.weight,
                    device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                    current_placement=Replicate(),
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
