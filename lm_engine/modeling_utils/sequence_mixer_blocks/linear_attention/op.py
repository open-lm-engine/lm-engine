# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....utils import divide_if_divisible


def linear_attention_torch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h0: torch.Tensor | None, attention_multiplier: float
) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, Nq, K = q.size()
    Nk = k.size(-2)
    Nv = v.size(-2)
    N = max(Nq, Nk, Nv)

    Gq = divide_if_divisible(N, Nq)
    Gk = divide_if_divisible(N, Nk)
    Gv = divide_if_divisible(N, Nv)

    y_shape = list(v.size())
    y_shape[-2] = N
    y = torch.empty(y_shape, device=q.device, dtype=q.dtype)

    V = v.size(-1)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    q = q.repeat_interleave(Gq, dim=-2)
    k = k.repeat_interleave(Gk, dim=-2)
    v = v.repeat_interleave(Gv, dim=-2)

    h0 = torch.zeros(B, N, K, V, dtype=torch.float32, device=q.device) if h0 is None else h0.float()

    for s in range(S):
        y[:, s] = (q[:, s, :, None, :] @ h0.type_as(q)).squeeze(-2)
        h0 = h0 + k[:, s, ..., None] * v[:, s, :, None, :]

    y = y * attention_multiplier

    return y, h0
