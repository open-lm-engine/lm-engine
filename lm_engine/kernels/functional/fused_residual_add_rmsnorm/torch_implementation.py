# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F


def _fused_residual_add_rmsnorm_torch(
    x: torch.Tensor,
    r: torch.Tensor | None,
    W: torch.Tensor | None,
    eps: float | None,
    multiplier: float | None,
    memory_efficient: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if multiplier not in [None, 1]:
        x = x * multiplier

    if r is not None:
        x = x + r
        r = x

    x = F.rms_norm(x, normalized_shape=(x.size(-1),), weight=W, eps=eps)

    return x, r
