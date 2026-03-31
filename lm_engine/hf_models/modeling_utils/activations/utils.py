# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return F.sigmoid(x.float()).type_as(x)


def tanh(x: torch.Tensor) -> torch.Tensor:
    return F.tanh(x.float()).type_as(x)


class _ClipGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gradient_clipping: float) -> torch.Tensor:
        ctx.gradient_clipping = gradient_clipping
        return x

    @staticmethod
    def backward(ctx, x_grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        gradient_clipping = ctx.gradient_clipping
        x_grad = x_grad.clip(-gradient_clipping, gradient_clipping)
        return x_grad, None


def clip_gradients(x: torch.Tensor, gradient_clipping: float | None) -> torch.Tensor:
    return x if gradient_clipping is None else _ClipGradients.apply(x, gradient_clipping)
