# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....autotuner import AutotuneConfig, autotune
from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2
from .forward import _get_autotune_configs, _get_online_autotune_configs


@triton.autotune(configs=_get_autotune_configs(), key=[])
@triton.jit
def _softmax_backward_triton_kernel(
    y_ptr,
    y_stride,
    dy_ptr,
    dy_stride,
    dx_ptr,
    dx_stride,
    logits_multiplier,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)

    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H

    MASK_BH = MASK_B[:, None] & MASK_H[None, :]

    y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1]
    dy_ptrs = dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :] * dy_stride[1]

    y = tl.load(y_ptrs, mask=MASK_BH)
    dy = tl.load(dy_ptrs, mask=MASK_BH)

    accumulator = (dy * y).to(tl.float32)
    accumulator = tl.sum(accumulator, axis=1, keep_dims=True)

    dy -= accumulator
    y *= dy
    if logits_multiplier is not None:
        y *= logits_multiplier

    tl.store(dx_ptr + BLOCK_B[:, None] * dx_stride[0] + BLOCK_H[None, :] * dx_stride[1], y, mask=MASK_BH)


@triton.autotune(configs=_get_online_autotune_configs(), key=[])
@triton.jit
def _online_softmax_backward_triton_kernel(
    y_ptr,
    y_stride,
    dy_ptr,
    dy_stride,
    dx_ptr,
    dx_stride,
    logits_multiplier,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)

    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    MASK_B = BLOCK_B < B

    accumulator = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1]
    dy_ptrs = dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :] * dy_stride[1]

    for _ in range(NUM_BLOCKS_H):
        MASK_H = BLOCK_H < H
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]

        y = tl.load(y_ptrs, mask=MASK_BH)
        y_ptrs += BLOCK_SIZE_H * y_stride[1]

        dy = tl.load(dy_ptrs, mask=MASK_BH)
        dy_ptrs += BLOCK_SIZE_H * dy_stride[1]

        acc = dy * y
        acc = acc.to(tl.float32)
        accumulator += tl.sum(acc, axis=1, keep_dims=True)

        BLOCK_H += BLOCK_SIZE_H

    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1]
    dy_ptrs = dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :] * dy_stride[1]
    dx_ptrs = dx_ptr + BLOCK_B[:, None] * dx_stride[0] + BLOCK_H[None, :] * dx_stride[1]

    for _ in range(NUM_BLOCKS_H):
        MASK_H = BLOCK_H < H
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]

        y = tl.load(y_ptrs, mask=MASK_BH)
        y_ptrs += BLOCK_SIZE_H * y_stride[1]

        dy = tl.load(dy_ptrs, mask=MASK_BH)
        dy_ptrs += BLOCK_SIZE_H * dy_stride[1]

        dy -= accumulator
        y *= dy
        if logits_multiplier is not None:
            y *= logits_multiplier

        tl.store(dx_ptrs, y, mask=MASK_BH)
        dx_ptrs += BLOCK_SIZE_H * dx_stride[1]

        BLOCK_H += BLOCK_SIZE_H


@autotune(
    configs=[
        AutotuneConfig({"use_online_softmax": False}, condition=lambda **kwargs: kwargs["dx"].size(1) <= 1024),
        AutotuneConfig({"use_online_softmax": True}),
    ],
    functional_triggers={"H": lambda **kwargs: get_next_power_of_2(kwargs["dx"].size(1))},
)
def _autotuned_softmax_backward_triton(
    y: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor, logits_multiplier: float | None, use_online_softmax: bool
) -> None:
    B, H = dx.size()
    GRID = lambda kwargs: (ceil_divide(B, kwargs["BLOCK_SIZE_B"]),)

    kwargs = {
        "y_ptr": y,
        "y_stride": y.stride(),
        "dy_ptr": dy,
        "dy_stride": dy.stride(),
        "dx_ptr": dx,
        "dx_stride": dx.stride(),
        "logits_multiplier": logits_multiplier,
        "B": B,
        "H": H,
    }

    if use_online_softmax:
        _online_softmax_backward_triton_kernel[GRID](**kwargs)
    else:
        _softmax_backward_triton_kernel[GRID](**kwargs, BLOCK_SIZE_H=get_next_power_of_2(H))


@xma_op(mutates_args={"dx"})
def _softmax_backward_triton(
    y: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor, logits_multiplier: float | None
) -> None:
    _autotuned_softmax_backward_triton(y=y, dy=dy, dx=dx, logits_multiplier=logits_multiplier)
