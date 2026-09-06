# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....custom_op import ctx_save_for_backward
from ..utils import _get_num_heads
from .backward import _rnn_backward_triton
from .forward import _rnn_forward_triton


class _RNNTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W: torch.Tensor,
        h0: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Nx, _, N = _get_num_heads(x=x, W=W, run_check=False)
        y_shape = list(x.size())
        y_shape[-2] = N

        y = torch.empty(y_shape, device=x.device, dtype=x.dtype)

        _rnn_forward_triton(
            x=x,
            W=W,
            h0=h0,
            y=y,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        ctx_save_for_backward(ctx, W, y, h0, cu_seqlens)

        ctx.max_seqlen = max_seqlen
        ctx.gradient_clipping = gradient_clipping
        ctx.Nx = Nx

        ht = y[:, -1] if cu_seqlens is None else y[cu_seqlens[1:] - 1]
        ht = ht.detach()

        return y, ht

    @staticmethod
    def backward(
        ctx, dy: torch.Tensor, dht: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, None, None]:
        W, y, h0, cu_seqlens = ctx.saved_tensors
        Nx = ctx.Nx
        N = y.size(-2)

        if Nx == N:
            dx = torch.empty_like(y, memory_format=torch.contiguous_format)
        else:
            x_shape = list(y.size())
            x_shape[-2] = Nx
            dx = torch.zeros(x_shape, device=y.device, dtype=torch.float32)

        dW = torch.zeros_like(W, dtype=torch.float32, memory_format=torch.contiguous_format)
        dh0 = (
            torch.empty_like(h0, memory_format=torch.contiguous_format)
            if h0 is not None and h0.requires_grad
            else None
        )

        _rnn_backward_triton(
            W=W,
            y=y,
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
        dW = dW.type_as(W)

        return dx, dW, dh0, None, None, None
