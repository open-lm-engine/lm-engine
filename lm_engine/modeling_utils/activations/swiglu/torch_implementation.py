# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F


def _swiglu_packed_torch(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()

    u = x[..., 1::2]
    g = x[..., ::2]

    x = u * F.silu(g)

    return x.to(dtype)
