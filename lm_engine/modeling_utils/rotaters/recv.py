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
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        grad_output = grad_output.contiguous()
        _send_op(x=grad_output, shift=-ctx.shift)

        return None, None, None, None


def recv(x: torch.Tensor, shift: int = 1) -> torch.Tensor:
    return _Recv.apply(x, shift)
