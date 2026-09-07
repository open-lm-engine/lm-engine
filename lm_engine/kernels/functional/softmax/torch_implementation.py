# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F


def _softmax_torch(x: torch.Tensor, logits_multiplier: float | None) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()

    if logits_multiplier is not None:
        x = x * logits_multiplier

    x = F.softmax(x, dim=-1)
    x = x.to(dtype)

    return x
