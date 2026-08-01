# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .send import _recv_op, _send_op


class _Recv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dummy: torch.Tensor, shape: torch.Size, dtype: torch.dtype, shift: int) -> torch.Tensor:
        ctx.shift = shift

        y = torch.empty(shape, dtype=dtype, device=dummy.device)
        _recv_op(y=y, shift=shift)

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        grad_output = grad_output.contiguous()
        _send_op(x=grad_output, shift=-ctx.shift)

        return None, None, None, None


def recv(shape: torch.Size, dtype: torch.dtype, device: torch.device, shift: int = 1) -> torch.Tensor:
    dummy = torch.empty(0, device=device, requires_grad=True)
    return _Recv.apply(dummy, shape, dtype, shift)
