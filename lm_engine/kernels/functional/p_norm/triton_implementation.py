# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...custom_op import xma_op
from ...math import get_next_power_of_2, get_powers_of_2
from ...triton_utils import compute_p_norm


@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for num_warps in get_powers_of_2(2, 16)], key=[])
@triton.jit
def _p_norm_triton_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
    multiplier,
    H: tl.constexpr,
    eps,
    is_P_inf: tl.constexpr,
    P: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    P_inv: tl.constexpr = None if is_P_inf else 1 / P

    BLOCK_ID_B = tl.program_id(0)

    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    MASK_H = BLOCK_H < H

    x = tl.load(x_ptr + BLOCK_ID_B * x_stride[0] + BLOCK_H * x_stride[1], mask=MASK_H)

    if multiplier is not None:
        x *= multiplier

    y = compute_p_norm(x=x, P=P, P_inv=P_inv, is_P_inf=is_P_inf, eps=eps, axis=0)
    tl.store(y_ptr + BLOCK_ID_B * y_stride[0], y)


@xma_op(mutates_args={"y"})
def _p_norm_triton(
    x: torch.Tensor, y: torch.Tensor, multiplier: float | None, p: int | None, is_p_inf: bool, eps: float
) -> None:
    B, H = x.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

    _p_norm_triton_kernel[B,](
        x_ptr=x,
        x_stride=x.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        multiplier=multiplier,
        H=H,
        eps=eps,
        is_P_inf=is_p_inf,
        P=p,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
