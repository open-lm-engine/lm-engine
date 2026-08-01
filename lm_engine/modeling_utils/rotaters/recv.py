# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .send import _recv_op, _send_op


class _Recv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, shift: int) -> torch.Tensor:
        ctx.shift = shift

        _recv_op(y=x, shift=shift)

        return x

    @staticmethod
    def backward(ctx, dx: torch.Tensor) -> tuple:
        dx = dx.contiguous()
        _send_op(x=dx, shift=-ctx.shift)

        return dx, None


def recv(x: torch.Tensor, shift: int = 1) -> torch.Tensor:
    return _Recv.apply(x, shift)
