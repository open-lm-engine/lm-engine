# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....custom_op import ctx_needs_gradients, ctx_save_for_backward
from ..utils import _get_backward_tensor, _get_num_heads
from .backward import _gru_backward_triton
from .forward import _gru_forward_triton


class _GRUTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        Nx, Nxf, Nxr, _, _, _, N = _get_num_heads(x=x, W=W, xf=xf, Wf=Wf, xr=xr, Wr=Wr, run_check=False)
        y_shape = list(x.size())
        y_shape[-2] = N

        needs_grad = ctx_needs_gradients(ctx)

        y = torch.empty(y_shape, device=x.device, dtype=x.dtype)
        f = torch.empty(y_shape, device=x.device, dtype=x.dtype) if needs_grad and Nxf == N else None
        r = torch.empty(y_shape, device=x.device, dtype=x.dtype) if needs_grad and Nxr == N else None
        z = torch.empty(y_shape, device=x.device, dtype=x.dtype) if needs_grad and Nx == N else None

        _gru_forward_triton(
            x=x,
            W=W,
            xf=xf,
            Wf=Wf,
            f=f,
            xr=xr,
            Wr=Wr,
            r=r,
            z=z,
            h0=h0,
            y=y,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        ctx_save_for_backward(
            ctx,
            W,
            Wf,
            f,
            Wr,
            r,
            z,
            y,
            h0,
            cu_seqlens,
            x if z is None else None,
            xf if f is None else None,
            xr if r is None else None,
        )

        ctx.max_seqlen = max_seqlen
        ctx.gradient_clipping = gradient_clipping
        ctx.num_heads = Nx, Nxf, Nxr

        ht = y[:, -1] if cu_seqlens is None else y[cu_seqlens[1:] - 1]
        ht = ht.detach()

        return y, ht

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: torch.Tensor | None):
        W, Wf, f, Wr, r, z, y, h0, cu_seqlens, x, xf, xr = ctx.saved_tensors
        Nx, Nxf, Nxr = ctx.num_heads

        dx = _get_backward_tensor(y=y, Nx=Nx, N=y.size(-2))
        dxf = _get_backward_tensor(y=y, Nx=Nxf, N=y.size(-2))
        dxr = _get_backward_tensor(y=y, Nx=Nxr, N=y.size(-2))

        dW = torch.zeros_like(W, dtype=torch.float32, memory_format=torch.contiguous_format)
        dWf = torch.zeros_like(Wf, dtype=torch.float32, memory_format=torch.contiguous_format)
        dWr = torch.zeros_like(Wr, dtype=torch.float32, memory_format=torch.contiguous_format)

        dh0 = (
            torch.empty_like(h0, memory_format=torch.contiguous_format)
            if h0 is not None and h0.requires_grad
            else None
        )

        _gru_backward_triton(
            x=x,
            W=W,
            y=y,
            xf=xf,
            Wf=Wf,
            f=f,
            dxf=dxf,
            dWf=dWf,
            xr=xr,
            Wr=Wr,
            r=r,
            dxr=dxr,
            dWr=dWr,
            z=z,
            h0=h0,
            dy=dy,
            dht=dht,
            dx=dx,
            dW=dW,
            dh0=dh0,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            gradient_clipping=ctx.gradient_clipping,
        )

        dx = dx.type_as(y)
        dxf = dxf.type_as(y)
        dxr = dxr.type_as(y)

        dW = dW.type_as(W)
        dWf = dWf.type_as(Wf)
        dWr = dWr.type_as(Wr)

        return dx, dW, dxf, dWf, dxr, dWr, dh0, None, None, None
