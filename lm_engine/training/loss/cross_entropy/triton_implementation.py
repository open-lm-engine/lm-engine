# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ...custom_op import ctx_needs_gradients, ctx_save_for_backward, xma_op
from ...math import ceil_divide, get_next_power_of_2, get_powers_of_2


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_SIZE_B in get_powers_of_2(1, 8):
        for num_warps in get_powers_of_2(4, 8):
            configs.append(triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B}, num_warps=num_warps))

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_V"], reset_to_zero=["l_ptr"])
@triton.jit
def _cross_entropy_forward_backward_triton_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
    l_ptr,
    dx_ptr,
    dx_stride,
    logits_multiplier,
    B,
    V,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    reduction: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)

    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    MASK_B = BLOCK_B < B

    Z = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    M = tl.full((BLOCK_SIZE_B, 1), -float("inf"), dtype=tl.float32)

    NUM_BLOCKS_V = tl.cdiv(V, BLOCK_SIZE_V)
    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)
    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_V[None, :] * x_stride[1]

    for _ in range(NUM_BLOCKS_V):
        MASK_V = BLOCK_V < V
        MASK_BV = MASK_B[:, None] & MASK_V[None, :]

        x = tl.load(x_ptrs, mask=MASK_BV, other=-float("inf")).to(tl.float32)
        x_ptrs += BLOCK_SIZE_V * x_stride[1]

        if logits_multiplier is not None:
            x *= logits_multiplier

        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)

        x -= M
        x = tl.exp(x)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)

        BLOCK_V += BLOCK_SIZE_V

    y = tl.load(y_ptr + BLOCK_B * y_stride[0], mask=MASK_B)

    xy = tl.load(x_ptr + BLOCK_B * x_stride[0] + y * x_stride[1], mask=MASK_B).to(tl.float32)
    if logits_multiplier is not None:
        xy *= logits_multiplier

    l = M + tl.log(Z) - xy[:, None]
    l = tl.where(MASK_B[:, None], l, 0)
    l = tl.sum(l, axis=0)

    if reduction == "mean":
        l /= B

    tl.atomic_add(l_ptr + tl.arange(0, 1), l, sem="relaxed")

    if dx_ptr is not None:
        BLOCK_V = tl.arange(0, BLOCK_SIZE_V)
        x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_V[None, :] * x_stride[1]
        dx_ptrs = dx_ptr + BLOCK_B[:, None] * dx_stride[0] + BLOCK_V[None, :] * dx_stride[1]

        for _ in range(NUM_BLOCKS_V):
            MASK_V = BLOCK_V < V
            MASK_BV = MASK_B[:, None] & MASK_V[None, :]

            x = tl.load(x_ptrs, mask=MASK_BV).to(tl.float32)
            x_ptrs += BLOCK_SIZE_V * x_stride[1]

            if logits_multiplier is not None:
                x *= logits_multiplier

            x -= M
            x = tl.exp(x)
            x /= Z

            x -= tl.where(BLOCK_V[None, :] == y[:, None], 1, 0)

            if logits_multiplier is not None:
                x *= logits_multiplier
            if reduction == "mean":
                x /= B

            tl.store(dx_ptrs, x, mask=MASK_BV)
            dx_ptrs += BLOCK_SIZE_V * dx_stride[1]

            BLOCK_V += BLOCK_SIZE_V


@xma_op(mutates_args={"loss", "dx"})
def _cross_entropy_forward_backward_triton(
    x: torch.Tensor,
    labels: torch.Tensor,
    loss: torch.Tensor,
    dx: torch.Tensor | None,
    logits_multiplier: float | None,
    reduction: str,
) -> None:
    B, V = x.size()

    BLOCK_SIZE_V = min(get_next_power_of_2(V), 4096 if x.dtype == torch.float32 else 8192)
    GRID = lambda kwargs: (ceil_divide(B, kwargs["BLOCK_SIZE_B"]),)

    _cross_entropy_forward_backward_triton_kernel[GRID](
        x_ptr=x,
        x_stride=x.stride(),
        y_ptr=labels,
        y_stride=labels.stride(),
        l_ptr=loss,
        dx_ptr=dx,
        dx_stride=None if dx is None else dx.stride(),
        logits_multiplier=logits_multiplier,
        B=B,
        V=V,
        reduction=reduction,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )


class _CrossEntropyTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, labels: torch.Tensor, reduction: str, logits_multiplier: float | None
    ) -> torch.Tensor:
        loss = torch.zeros((), device=x.device, dtype=torch.float32)
        dx = torch.empty_like(x, memory_format=torch.contiguous_format) if ctx_needs_gradients(ctx) else None

        _cross_entropy_forward_backward_triton(
            x=x, labels=labels, loss=loss, dx=dx, logits_multiplier=logits_multiplier, reduction=reduction
        )

        ctx_save_for_backward(ctx, dx)

        return loss

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        dx = ctx.saved_tensors[0]
        dx *= dy

        return dx, None, None, None
