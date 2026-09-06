# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...accelerator import KernelBackend
from ..cross_entropy import cross_entropy


def _fused_linear_cross_entropy_torch(
    x: torch.Tensor,
    W: torch.Tensor,
    y: torch.Tensor,
    reduction: str,
    logits_multiplier: float | None,
) -> torch.Tensor:
    x = F.linear(x, W)
    l = cross_entropy(
        x=x, labels=y, reduction=reduction, logits_multiplier=logits_multiplier, kernel_backend=KernelBackend.torch
    )

    return l
