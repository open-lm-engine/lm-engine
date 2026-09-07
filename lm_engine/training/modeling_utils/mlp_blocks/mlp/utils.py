# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch


def interleave_up_gate_tensor_for_mlp(
    up_weight: torch.Tensor, gate_weight: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    if dim == 0:
        W = torch.empty(2 * up_weight.size(0), *up_weight.size()[1:], dtype=up_weight.dtype, device=up_weight.device)
        W[1::2] = up_weight
        W[::2] = gate_weight
    elif dim == 1:
        W = torch.empty(
            up_weight.size(0),
            2 * up_weight.size(1),
            *up_weight.size()[2:],
            dtype=up_weight.dtype,
            device=up_weight.device,
        )
        W[:, 1::2] = up_weight
        W[:, ::2] = gate_weight
    else:
        raise ValueError("dim >= 2 is not supported")

    return W


def split_up_gate_tensor_for_mlp(c_fc_weight: torch.Tensor, dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if dim == 0:
        u = c_fc_weight[1::2].contiguous()
        g = c_fc_weight[::2].contiguous()
    elif dim == 1:
        u = c_fc_weight[:, 1::2].contiguous()
        g = c_fc_weight[:, ::2].contiguous()
    else:
        raise ValueError(f"Unsupported dim: {dim}. Only dim=0 or dim=1 are supported.")

    return u, g
