# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...custom_op import ctx_needs_gradients, ctx_save_for_backward
from ...math import ceil_divide, get_next_power_of_2
from ..cross_entropy.triton_implementation import _cross_entropy_forward_backward_triton


class _FusedLinearCrossEntropyTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W: torch.Tensor,
        y: torch.Tensor,
        reduction: str,
        logits_multiplier: float | None,
    ) -> torch.Tensor:
        B, H = x.size()
        V = W.size(0)

        # NOTE chunking is copied from liger kernel
        memory_increase_factor = ceil_divide(V, H)
        # chunk_size needed to reduce memory increase back to 1
        chunk_size = get_next_power_of_2(ceil_divide(B, memory_increase_factor))
        num_chunks = ceil_divide(B, chunk_size)

        l = torch.zeros((), device=x.device, dtype=torch.float32)

        needs_grad = ctx_needs_gradients(ctx)
        dx = torch.empty_like(x, memory_format=torch.contiguous_format) if needs_grad else None
        dW = torch.zeros_like(W, memory_format=torch.contiguous_format) if needs_grad else None

        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            end = min(end, B)

            _x = x[start:end]
            _h = _x @ W.T

            _dh = torch.empty_like(_h, memory_format=torch.contiguous_format)
            _y = y[start:end]

            _cross_entropy_forward_backward_triton(
                x=_h, labels=_y, loss=l, dx=_dh, logits_multiplier=logits_multiplier, reduction="sum"
            )

            if needs_grad:
                dx[start:end] = _dh @ W
                torch.addmm(dW, _dh.T, _x, alpha=1, beta=1, out=dW)

        if reduction == "mean":
            l /= B
            dx /= B
            dW /= B

        ctx_save_for_backward(ctx, dx, dW)

        return l

    @staticmethod
    def backward(ctx, dl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, None, None, None]:
        dx, dW = ctx.saved_tensors

        dx *= dl
        dW *= dl

        return dx, dW, None, None, None
