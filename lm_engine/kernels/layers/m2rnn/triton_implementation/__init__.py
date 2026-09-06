# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import torch

from ....custom_op import ctx_save_for_backward
from ..utils import _get_num_heads
from .backward import _m2rnn_backward_triton
from .forward import _MAX_BLOCK_SIZE_K, _m2rnn_forward_triton


class _M2RNNTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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

        if cu_seqlens is None:
            B = k.size(0)
        else:
            B = cu_seqlens.size(0) - 1

        K = k.size(-1)
        V = v.size(-1)

        ht = torch.empty(B, N, K, V, device=k.device, dtype=k.dtype)

        y_shape = list(v.size())
        y_shape[-2] = N

        if K > _MAX_BLOCK_SIZE_K:
            y = torch.zeros(y_shape, device=q.device, dtype=torch.float32)
        else:
            y = torch.empty(y_shape, device=q.device, dtype=q.dtype)

        _m2rnn_forward_triton(
            q=q,
            k=k,
            v=v,
            W=W,
            xf=xf,
            h0=h0,
            h=None,
            ht=ht,
            y=y,
            cu_seqlens=cu_seqlens,
            Nq=Nq,
            Nk=Nk,
            Nv=Nv,
            Nw=Nw,
            Nxf=Nxf,
            N=N,
        )

        y = y.type_as(v)

        ctx_save_for_backward(ctx, q, k, v, W, xf, h0, cu_seqlens)
        ctx.gradient_clipping = gradient_clipping
        ctx.num_heads = Nq, Nk, Nv, Nw, Nxf, N

        return y, ht

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: torch.Tensor):
        q, k, v, W, xf, h0, cu_seqlens = ctx.saved_tensors
        Nq, Nk, Nv, Nw, Nxf, N = ctx.num_heads

        V = v.size(-1)

        if cu_seqlens is None:
            B, S, _, K = q.size()
            h = torch.empty(B, S, N, K, V, dtype=q.dtype, device=q.device)
        else:
            T, _, K = q.size()
            h = torch.empty(T, N, K, V, dtype=q.dtype, device=q.device)

        _m2rnn_forward_triton(
            q=None,
            k=k,
            v=v,
            W=W,
            xf=xf,
            h0=h0,
            h=h,
            ht=None,
            y=None,
            cu_seqlens=cu_seqlens,
            Nq=Nq,
            Nk=Nk,
            Nv=Nv,
            Nw=Nw,
            Nxf=Nxf,
            N=N,
        )

        empty = partial(torch.empty_like, memory_format=torch.contiguous_format)
        function = partial(torch.zeros_like, dtype=torch.float32, memory_format=torch.contiguous_format)

        dq = (empty if Nq == N else function)(q)
        dk = (empty if Nk == N else function)(k)
        dW = torch.zeros_like(W, dtype=torch.float32, memory_format=torch.contiguous_format)
        dh0 = empty(h0) if h0 is not None and h0.requires_grad else None

        if K > _MAX_BLOCK_SIZE_K:
            dv = function(v)
            dxf = function(xf)
        else:
            dv = (empty if Nv == N else function)(v)
            dxf = (empty if Nxf == N else function)(xf)

        _m2rnn_backward_triton(
            q=q,
            k=k,
            v=v,
            W=W,
            xf=xf,
            h0=h0,
            dy=dy,
            dht=dht,
            h=h,
            dq=dq,
            dk=dk,
            dv=dv,
            dW=dW,
            dxf=dxf,
            dh0=dh0,
            cu_seqlens=cu_seqlens,
            gradient_clipping=ctx.gradient_clipping,
        )

        dq = dq.type_as(q)
        dk = dk.type_as(k)
        dv = dv.type_as(v)
        dW = dW.type_as(W)
        dxf = dxf.type_as(xf)

        return dq, dk, dv, dW, dxf, dh0, None, None, None
