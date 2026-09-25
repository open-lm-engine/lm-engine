# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from .dtensor_module import DTensorModule


class Dropout(nn.Dropout, DTensorModule):
    def __init__(
        self, p: float = 0.5, use_padding_free_transformer: bool = False, sequence_parallel: bool = False
    ) -> Dropout:
        super().__init__(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # early exit
        if self.p == 0:
            return x

        return super().forward(x)
