# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from ...activations import clip_gradients, tanh


def rnn_torch(
    x: torch.Tensor, state_weight: torch.Tensor, h0: torch.Tensor | None, gradient_clipping: float | None
) -> tuple[torch.Tensor, torch.Tensor]:
    W = state_weight

    Nx = x.size(-2)
    Nw = W.size(0)
    N = max(Nx, Nw)

    y_shape = list(x.size())
    y_shape[-2] = N
    y = torch.empty(y_shape, device=x.device, dtype=x.dtype)

    B, S, _, H = x.size()
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
