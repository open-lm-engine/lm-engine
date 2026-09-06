# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...activations import clip_gradients, sigmoid, tanh
from .utils import _get_num_heads


def _gru_torch(
    x: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    Wf: torch.Tensor,
    xr: torch.Tensor,
    Wr: torch.Tensor,
    h0: torch.Tensor | None,
    gradient_clipping: float | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    Nx, Nxf, Nxr, Nw, Nwf, Nwr, N = _get_num_heads(x=x, W=W, xf=xf, Wf=Wf, xr=xr, Wr=Wr, run_check=False)

    y_shape = list(x.size())
    y_shape[-2] = N
    y = torch.empty(y_shape, device=x.device, dtype=x.dtype)

    if cu_seqlens is None:
        B, S, _, H = x.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = max_seqlen
        H = x.size(-1)

    Gx = N // Nx
    Gxf = N // Nxf
    Gxr = N // Nxr

    Gw = N // Nw
    Gwf = N // Nwf
    Gwr = N // Nwr

    x = x.repeat_interleave(Gx, dim=-2)
    xf = xf.repeat_interleave(Gxf, dim=-2)
    xr = xr.repeat_interleave(Gxr, dim=-2)

    W = W.repeat_interleave(Gw, dim=0)[None, ...]
    Wf = Wf.repeat_interleave(Gwf, dim=0)[None, ...]
    Wr = Wr.repeat_interleave(Gwr, dim=0)[None, ...]

    if h0 is None:
        h0 = torch.zeros(B, N, H, device=x.device, dtype=x.dtype)

    if cu_seqlens is not None:
        h0 = h0.clone()
        start = cu_seqlens[:-1]
        end = cu_seqlens[1:]

    for s in range(S):
        if cu_seqlens is None:
            f = h0[..., None, :] @ Wf + xf[:, s, :, None, :]
            r = h0[..., None, :] @ Wr + xr[:, s, :, None, :]
        else:
            offset = start + s
            unfinished = offset < end
            offset_unfinished = offset[unfinished]

            f = h0[unfinished, :, None, :] @ Wf + xf[offset_unfinished, :, None, :]
            r = h0[unfinished, :, None, :] @ Wr + xr[offset_unfinished, :, None, :]

        f = sigmoid(f)
        r = sigmoid(r)

        if cu_seqlens is None:
            z = (h0[..., None, :] * r) @ W + x[:, s, :, None, :]
        else:
            z = (h0[unfinished, :, None, :] * r) @ W + x[offset_unfinished, :, None, :]

        z = tanh(z)

        if cu_seqlens is None:
            h = f * h0[..., None, :] + (1 - f) * z
        else:
            h = f * h0[unfinished, :, None, :] + (1 - f) * z

        h = h.squeeze(-2)
        h = clip_gradients(h, gradient_clipping)

        if cu_seqlens is None:
            y[:, s] = h
            h0 = h
        else:
            y[offset_unfinished] = h
            h0[unfinished] = h

    return y, h0
