# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...torch_utils import clip_gradients, sigmoid, tanh
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
        raise NotImplementedError

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

    for s in range(S):
        f = h0[..., None, :] @ Wf + xf[:, s, :, None, :]
        r = h0[..., None, :] @ Wr + xr[:, s, :, None, :]

        f = sigmoid(f)
        r = sigmoid(r)

        z = (h0[..., None, :] * r) @ W + x[:, s, :, None, :]
        z = tanh(z)
        h = f * h0[..., None, :] + (1 - f) * z

        h = h.squeeze(-2)
        h = clip_gradients(h, gradient_clipping)

        y[:, s] = h
        h0 = h

    return y, h0
