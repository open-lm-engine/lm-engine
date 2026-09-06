# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import MAX_TRITON_BLOCK_SIZE
from ....custom_op import xma_op
from ....math import get_next_power_of_2, get_powers_of_2


@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for num_warps in get_powers_of_2(2, 16)], key=[])
@triton.jit
def _fused_residual_add_rmsnorm_forward_triton_kernel(
    x_ptr,
    x_stride,
    r_ptr,
    r_stride,
    W_ptr,
    W_stride,
    y_ptr,
    y_stride,
    xr_ptr,
    xr_stride,
    s_ptr,
    s_stride,
    eps,
    multiplier,
    B,
    H: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    H_inv: tl.constexpr = 1 / H

    BLOCK_ID_B = tl.program_id(0)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H

    MASK_BH = MASK_B[:, None] & MASK_H[None, :]

    x = tl.load(x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :] * x_stride[1], mask=MASK_BH).to(tl.float32)

    if multiplier is not None:
        x *= multiplier

    if r_ptr is not None:
        r = tl.load(r_ptr + BLOCK_B[:, None] * r_stride[0] + BLOCK_H[None, :] * r_stride[1], mask=MASK_BH)
        x += r

    if xr_ptr is not None:
        tl.store(xr_ptr + BLOCK_B[:, None] * xr_stride[0] + BLOCK_H[None, :] * xr_stride[1], x, mask=MASK_BH)

    r = tl.sum(x * x, axis=1)
    r = tl.rsqrt(r * H_inv + eps)

    if s_ptr is not None:
        tl.store(s_ptr + BLOCK_B * s_stride[0], r, mask=MASK_B)

    x *= r[:, None]

    if W_ptr is not None:
        W = tl.load(W_ptr + BLOCK_H * W_stride[0], mask=MASK_H)
        x = x.to(x_ptr.dtype.element_ty) * W[None, :]

    tl.store(y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1], x, mask=MASK_BH)


@xma_op(mutates_args={"y", "xr", "s"})
def _fused_residual_add_rmsnorm_forward_triton(
    x: torch.Tensor,
    r: torch.Tensor | None,
    W: torch.Tensor | None,
    y: torch.Tensor,
    eps: float,
    multiplier: float | None,
    xr: torch.Tensor | None,
    s: torch.Tensor | None,
) -> None:
    B, H = x.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

    _fused_residual_add_rmsnorm_forward_triton_kernel[B,](
        x_ptr=x,
        x_stride=x.stride(),
        r_ptr=r,
        r_stride=None if r is None else r.stride(),
        W_ptr=W,
        W_stride=None if W is None else W.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        xr_ptr=xr,
        xr_stride=None if xr is None else xr.stride(),
        s_ptr=s,
        s_stride=None if s is None else s.stride(),
        eps=eps,
        multiplier=multiplier,
        B=B,
        H=H,
        BLOCK_SIZE_B=1,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
