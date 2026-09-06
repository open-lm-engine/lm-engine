# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Callable

import torch

from ...accelerator import Accelerator


def _forward_function(
    x: torch.Tensor,
    r: torch.Tensor | None,
    W: torch.Tensor | None,
    eps: float | None,
    multiplier: float | None,
    output_std: bool,
    forward_op: Callable,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if eps is None:
        eps = torch.finfo(x.dtype).eps

    B = x.size(0)

    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    xr = None if r is None else torch.empty_like(x, memory_format=torch.contiguous_format)
    s = torch.empty(B, device=x.device, dtype=torch.float32) if output_std else None

    forward_op(x=x, r=r, W=W, y=y, eps=eps, multiplier=multiplier, xr=xr, s=s)

    return y, xr, s


def _backward_function(
    xr: torch.Tensor,
    W: torch.Tensor | None,
    s: torch.Tensor | None,
    dy: torch.Tensor,
    dxr: torch.Tensor | None,
    has_residual: bool,
    multiplier: float | None,
    eps: float | None,
    backward_op: Callable,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    dx = torch.empty_like(xr, memory_format=torch.contiguous_format)
    dr = torch.empty_like(xr, memory_format=torch.contiguous_format) if has_residual else None

    dW = (
        None
        if W is None
        else torch.empty(Accelerator.get_core_count(), *W.size(), dtype=torch.float32, device=dx.device)
    )

    if not has_residual:
        assert dxr is None

    if eps is None:
        eps = torch.finfo(dy.dtype).eps

    backward_op(
        xr=xr,
        W=W,
        dy=dy,
        dxr=dxr,
        s=s,
        dx=dx,
        dr=dr,
        dW=dW,
        eps=eps,
        multiplier=multiplier,
    )

    if dW is not None:
        dW = dW.sum(0)
        dW = dW.type_as(W)

    return dx, dr, dW
