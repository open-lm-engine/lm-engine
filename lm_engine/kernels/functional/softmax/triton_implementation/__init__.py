# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....custom_op import ctx_save_for_backward
from .backward import _softmax_backward_triton
from .forward import _softmax_forward_triton


class _SoftmaxTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, logits_multiplier: float | None) -> torch.Tensor:
        y = torch.empty_like(x, memory_format=torch.contiguous_format)

        _softmax_forward_triton(x=x, y=y, logits_multiplier=logits_multiplier)

        ctx_save_for_backward(ctx, y)
        ctx.logits_multiplier = logits_multiplier

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, None]:
        y = ctx.saved_tensors[0]
        dx = torch.empty_like(y, memory_format=torch.contiguous_format)

        _softmax_backward_triton(y=y, dy=dy, dx=dx, logits_multiplier=ctx.logits_multiplier)

        return dx, None
