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


def stitch_autograd_in_forward(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return _StitchAutogradInForward.apply(x, h)
