# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....autotuner import AutotuneConfig, autotune
from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2, get_powers_of_2


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_SIZE_B in get_powers_of_2(1, 64):
        for num_warps in get_powers_of_2(4, 32):
            configs.append(triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B}, num_warps=num_warps))

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=[])
@triton.jit
def _softmax_forward_triton_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
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

    x = tl.load(
        x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :] * x_stride[1], mask=MASK_BH, other=-float("inf")
    )

    x = x.to(tl.float32)

    if logits_multiplier is not None:
        x *= logits_multiplier

    x = tl.exp(x)
    x /= tl.sum(x, axis=1, keep_dims=True)

    tl.store(y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1], x, mask=MASK_BH)


def _get_online_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_SIZE_B in get_powers_of_2(1, 4):
        for BLOCK_SIZE_H in get_powers_of_2(16, 8192):
            for num_warps in get_powers_of_2(4, 32):
                configs.append(
                    triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B, "BLOCK_SIZE_H": BLOCK_SIZE_H}, num_warps=num_warps)
                )

    return configs


@triton.autotune(configs=_get_online_autotune_configs(), key=[])
@triton.jit
def _online_softmax_forward_triton_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
    logits_multiplier,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)

    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    MASK_B = BLOCK_B < B

    Z = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    M = tl.full((BLOCK_SIZE_B, 1), -float("inf"), dtype=tl.float32)

    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :] * x_stride[1]

    for _ in range(NUM_BLOCKS_H):
        MASK_H = BLOCK_H < H
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]

        x = tl.load(x_ptrs, mask=MASK_BH, other=-float("inf"))

        x = x.to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier

        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)

        x -= M
        x = tl.exp(x)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)

        BLOCK_H += BLOCK_SIZE_H
        x_ptrs += BLOCK_SIZE_H * x_stride[1]

    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :] * x_stride[1]
    y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1]

    for _ in range(NUM_BLOCKS_H):
        MASK_H = BLOCK_H < H
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]

        x = tl.load(x_ptrs, mask=MASK_BH)
        x_ptrs += BLOCK_SIZE_H * x_stride[1]

        x = x.to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier

        x -= M
        x = tl.exp(x)
        x /= Z

        tl.store(y_ptrs, x, mask=MASK_BH)
        y_ptrs += BLOCK_SIZE_H * y_stride[1]

        BLOCK_H += BLOCK_SIZE_H


@autotune(
    configs=[
        AutotuneConfig({"use_online_softmax": False}, condition=lambda **kwargs: kwargs["x"].size(1) <= 1024),
        AutotuneConfig({"use_online_softmax": True}),
    ],
    functional_triggers={"H": lambda **kwargs: get_next_power_of_2(kwargs["x"].size(1))},
)
def _autotuned_softmax_forward_triton(
    x: torch.Tensor, y: torch.Tensor, logits_multiplier: float | None, use_online_softmax: bool
) -> None:
    B, H = x.size()
    GRID = lambda kwargs: (ceil_divide(B, kwargs["BLOCK_SIZE_B"]),)

    kwargs = {
        "x_ptr": x,
        "x_stride": x.stride(),
        "y_ptr": y,
        "y_stride": y.stride(),
        "logits_multiplier": logits_multiplier,
        "B": B,
        "H": H,
    }

    if use_online_softmax:
        _online_softmax_forward_triton_kernel[GRID](**kwargs)
    else:
        _softmax_forward_triton_kernel[GRID](**kwargs, BLOCK_SIZE_H=get_next_power_of_2(H))


@xma_op(mutates_args={"y"})
def _softmax_forward_triton(x: torch.Tensor, y: torch.Tensor, logits_multiplier: float | None) -> None:
    _autotuned_softmax_forward_triton(x=x, y=y, logits_multiplier=logits_multiplier)
