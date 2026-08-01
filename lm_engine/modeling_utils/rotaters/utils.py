# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch


@torch.library.custom_op("lm_engine::_no_op", mutates_args={"x"})
def _no_op(x: torch.Tensor, h: torch.Tensor) -> None:
    return


class _StitchAutogradInForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ctx.shape = h.size()
        ctx.dtype = h.dtype
        ctx.device = h.device

        _no_op(x, h)

        return x

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return dy, torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)


class _StitchAutogradInBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, shape: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty(shape, dtype=dtype, device=device)

    @staticmethod
    def backward(ctx, dx: torch.Tensor, dh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _no_op(x=dx, h=dh)
        return dx, None, None, None


def stitch_autograd_in_forward(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return _StitchAutogradInForward.apply(x, h)


def stitch_autograd_in_backward(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return _StitchAutogradInBackward.apply(x, shape)
