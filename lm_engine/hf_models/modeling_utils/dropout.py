# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...dtensors import dtensor_to_tensor, tensor_to_dtensor
from ...utils import ProcessGroupManager
from .TP import get_module_placements


class Dropout(nn.Dropout):
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> Dropout:
        super().__init__(p, inplace)

        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()
        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.tp_world_size > 1:
                input = tensor_to_dtensor(input, device_mesh=self.tp_mesh, current_placement=self.placement)

            input = super().forward(input)

            if self.tp_world_size:
                input = dtensor_to_tensor(input, device_mesh=self.tp_mesh, desired_placement=self.placement)

        return input
