# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F
from torch.distributed.tensor import Replicate

from . import ParameterizedEmbedding
from .TP import get_tensor_parallel_activation_placements


class LMHead(ParameterizedEmbedding):
    def forward(self, x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        if weight is None:
            weight = self.weight

        return LMHead.compute_with_weight(x=x, weight=weight)

    @staticmethod
    def compute_with_weight(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = x.redistribute(
            device_mesh=weight.device_mesh,
            placements=get_tensor_parallel_activation_placements(Replicate()),
            async_op=True,
        )
        x = F.linear(x, weight)
        return x
