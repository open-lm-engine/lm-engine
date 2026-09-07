# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....custom_op import ctx_needs_gradients, ctx_save_for_backward
from ..template import _backward_function, _forward_function
from .backward import _fused_residual_add_rmsnorm_backward_triton
from .forward import _fused_residual_add_rmsnorm_forward_triton


class _FusedResidualAddRMSNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        r: torch.Tensor | None,
        W: torch.Tensor | None,
        eps: float | None,
        multiplier: float | None,
        memory_efficient: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        y, xr, s = _forward_function(
            x=x,
            r=r,
            W=W,
            eps=eps,
            multiplier=multiplier,
            output_std=ctx_needs_gradients(ctx) and not memory_efficient,
            forward_op=_fused_residual_add_rmsnorm_forward_triton,
        )

        has_residual = r is not None

        ctx_save_for_backward(ctx, xr if has_residual else x, W, s)
        ctx.eps = eps
        ctx.has_residual = has_residual
        ctx.multiplier = multiplier

        return y, xr

    @staticmethod
    def backward(
        ctx, dy: torch.Tensor, dxr: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, None, None, None]:
        xr, W, s = ctx.saved_tensors

        dx, dr, dW = _backward_function(
            xr=xr,
            W=W,
            s=s,
            dy=dy,
            dxr=dxr,
            has_residual=ctx.has_residual,
            multiplier=ctx.multiplier,
            eps=ctx.eps,
            backward_op=_fused_residual_add_rmsnorm_backward_triton,
        )

        return dx, dr, dW, None, None, None
