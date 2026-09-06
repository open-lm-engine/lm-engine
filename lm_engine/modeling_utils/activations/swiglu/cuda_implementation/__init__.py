# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .....custom_op import ctx_save_for_backward
from .....math import divide_if_divisible
from .backward import _swiglu_packed_backward_cuda
from .forward import _swiglu_packed_forward_cuda


class _SwigluPackedCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        y = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2), device=x.device, dtype=x.dtype)

        ctx_save_for_backward(ctx, x)
        _swiglu_packed_forward_cuda(x=x, y=y)

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]

        dy = dy.contiguous()
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)

        _swiglu_packed_backward_cuda(x=x, dy=dy, dx=dx)

        return dx
