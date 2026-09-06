# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...torch_utils import clip_gradients, tanh
from .utils import _get_num_heads


def _m2rnn_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    h0: torch.Tensor | None,
    gradient_clipping: float | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    Nq, Nk, Nv, Nw, Nxf, N = _get_num_heads(q=q, k=k, v=v, W=W, xf=xf, run_check=False)

    V = v.size(-1)

    if cu_seqlens is None:
        B, S, _, K = q.size()
        y = torch.empty(B, S, N, K, V, device=q.device, dtype=q.dtype)
    else:
        raise NotImplementedError

    if h0 is None:
        h0 = torch.zeros(B, N, K, V, device=k.device, dtype=k.dtype)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    Gw = N // Nw
    Gxf = N // Nxf

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
