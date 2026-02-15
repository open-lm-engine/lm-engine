# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...dtensors import dtensor_to_tensor, tensor_to_dtensor
from .dtensor_module import DTensorModule
from .TP import get_module_placements


class Dropout(nn.Dropout, DTensorModule):
    def __init__(
        self, p: float = 0.5, use_padding_free_transformer: bool = False, sequence_parallel: bool = False
    ) -> Dropout:
        super().__init__(p)

        if self.is_tp_enabled:
            self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # early exit
        if self.p == 0:
            return x

        if self.is_tp_enabled:
            x = tensor_to_dtensor(x, device_mesh=self.tp_mesh, current_placement=self.placement)
            x = super().forward(x)
            x = dtensor_to_tensor(x, device_mesh=self.tp_mesh, desired_placement=self.placement)
        else:
            x = super().forward(x)

        return x
