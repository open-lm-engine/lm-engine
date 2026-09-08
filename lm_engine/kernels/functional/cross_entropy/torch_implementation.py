# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F


def _cross_entropy_torch(
    x: torch.Tensor, labels: torch.Tensor, reduction: str = "mean", logits_multiplier: float | None = None
) -> torch.Tensor:
    x = x.float()

    if logits_multiplier not in [None, 1]:
        x = x * logits_multiplier

    return F.cross_entropy(x, labels, reduction=reduction)
