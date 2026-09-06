# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...torch_utils import clip_gradients, tanh
from .utils import _get_num_heads


def _rnn_torch(
    x: torch.Tensor,
    W: torch.Tensor,
    h0: torch.Tensor | None,
    gradient_clipping: float | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    Nx, Nw, N = _get_num_heads(x=x, W=W, run_check=False)

    y_shape = list(x.size())
    y_shape[-2] = N
    y = torch.empty(y_shape, device=x.device, dtype=x.dtype)

    if cu_seqlens is None:
        B, S, _, H = x.size()
    else:
        raise NotImplementedError

    Gx = N // Nx
    Gw = N // Nw

    x = x.repeat_interleave(Gx, dim=-2)
    W = W.repeat_interleave(Gw, dim=0)[None, ...]

    if h0 is None:
        h0 = torch.zeros(B, N, H, device=x.device, dtype=x.dtype)

    for s in range(S):
        h = h0[..., None, :] @ W + x[:, s, :, None, :]

        h = tanh(h)
        h = h.squeeze(-2)
        h = clip_gradients(h, gradient_clipping)

        y[:, s] = h
        h0 = h

    return y, h0
