# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....utils import divide_if_divisible
from ...activations import clip_gradients, tanh


def m2rnn_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    xf: torch.Tensor,
    W: torch.Tensor,
    h0: torch.Tensor | None,
    gradient_clipping: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, Nq, K = q.size()
    Nk = k.size(-2)
    Nv = v.size(-2)

    Nw = W.size(0)
    Nxf = xf.size(-1)

    N = max(Nq, Nk, Nv, Nw, Nxf)
    V = v.size(-1)
    y = torch.empty(B, S, N, K, V, device=q.device, dtype=q.dtype)

    if h0 is None:
        h0 = torch.zeros(B, N, K, V, device=k.device, dtype=k.dtype)

    Gq = divide_if_divisible(N, Nq)
    Gk = divide_if_divisible(N, Nk)
    Gv = divide_if_divisible(N, Nv)

    Gw = divide_if_divisible(N, Nw)
    Gxf = divide_if_divisible(N, Nxf)

    q = q.repeat_interleave(Gq, dim=-2)
    k = k.repeat_interleave(Gk, dim=-2)
    v = v.repeat_interleave(Gv, dim=-2)
    W = W.repeat_interleave(Gw, dim=0)
    xf = xf.repeat_interleave(Gxf, dim=-1)

    # (B, S, N, K, V) = (B, S, N, K, 1) * (B, S, N, 1, V)
    x = k[..., None] * v[..., None, :]
    W = W[None, ...]

    for s in range(S):
        f = xf[:, s, :, None, None]
        # (B, N, K, V) = (B, N, K, V) @ (1, N, V, V) + (B, N, K, V)
        h = h0 @ W + x[:, s]
        h = tanh(h)
        h = f * h0 + (1 - f) * h

        h = clip_gradients(h, gradient_clipping)

        y[:, s] = h
        h0 = h

    y = q[..., None, :] @ y
    y = y.squeeze(-2)

    return y, h0
