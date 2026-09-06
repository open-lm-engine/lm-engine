# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....accelerator import Accelerator
from ....constants import MAX_TRITON_BLOCK_SIZE
from ....custom_op import xma_op
from ....math import get_next_power_of_2, get_powers_of_2


@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for num_warps in get_powers_of_2(2, 16)], key=[])
@triton.jit
def _fused_residual_add_rmsnorm_backward_triton_kernel(
    xr_ptr,
    xr_stride,
    W_ptr,
    W_stride,
    dy_ptr,
    dy_stride,
    dxr_ptr,
    dxr_stride,
    dx_ptr,
    dx_stride,
    dr_ptr,
    dr_stride,
    dW_ptr,
    dW_stride,
    s_ptr,
    s_stride,
    eps,
    multiplier,
    B,
    H: tl.constexpr,
    H_inv,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)
    NUM_BLOCKS = tl.num_programs(0)

    NUM_ELEMENTS_PER_BLOCK = tl.cdiv(B, NUM_BLOCKS)

    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    MASK_H = BLOCK_H < H

    start = BLOCK_ID * NUM_ELEMENTS_PER_BLOCK
    end = min(start + NUM_ELEMENTS_PER_BLOCK, B)

    if W_ptr is not None:
        W = tl.load(W_ptr + BLOCK_H * W_stride[0], mask=MASK_H)
        dW = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)

    for BLOCK_ID_B in range(start, end):
        xr = tl.load(xr_ptr + BLOCK_ID_B * xr_stride[0] + BLOCK_H * xr_stride[1], mask=MASK_H).to(tl.float32)

        if s_ptr is None:
            r = tl.sum(xr * xr)
            r = tl.rsqrt(r * H_inv + eps)
        else:
            r = tl.load(s_ptr + BLOCK_ID_B * s_stride[0])

        z = r * xr

        dy = tl.load(dy_ptr + BLOCK_ID_B * dy_stride[0] + BLOCK_H * dy_stride[1], mask=MASK_H).to(tl.float32)

        dyW = dy
        if W_ptr is not None:
            dyW *= W
            dW += dy * z

        dx = (dyW - H_inv * z * tl.sum(dyW * z, keep_dims=True)) * r

        if dxr_ptr is not None:
            dx += tl.load(dxr_ptr + BLOCK_ID_B * dxr_stride[0] + BLOCK_H * dxr_stride[1], mask=MASK_H)

        if dr_ptr is not None:
            tl.store(dr_ptr + BLOCK_ID_B * dr_stride[0] + BLOCK_H * dr_stride[1], dx, mask=MASK_H)

        if multiplier is not None:
            dx *= multiplier

        tl.store(dx_ptr + BLOCK_ID_B * dx_stride[0] + BLOCK_H * dx_stride[1], dx, mask=MASK_H)

    if W_ptr is not None:
        tl.store(dW_ptr + BLOCK_ID * dW_stride[0] + BLOCK_H * dW_stride[1], dW, mask=MASK_H)


@xma_op(mutates_args={"dx", "dr", "dW"})
def _fused_residual_add_rmsnorm_backward_triton(
    xr: torch.Tensor,
    W: torch.Tensor | None,
    dy: torch.Tensor,
    dxr: torch.Tensor | None,
    s: torch.Tensor | None,
    dx: torch.Tensor,
    dr: torch.Tensor | None,
    dW: torch.Tensor | None,
    eps: float,
    multiplier: float | None,
) -> None:
    B, H = xr.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

    if dW is None:
        cores = Accelerator.get_core_count()
        GRID = lambda kwargs: (min(cores, B),)
    else:
        GRID = (dW.size(0),)

    _fused_residual_add_rmsnorm_backward_triton_kernel[GRID](
        xr_ptr=xr,
        xr_stride=None if xr is None else xr.stride(),
        W_ptr=W,
        W_stride=None if W is None else W.stride(),
        dy_ptr=dy,
        dy_stride=dy.stride(),
        dxr_ptr=dxr,
        dxr_stride=None if dxr is None else dxr.stride(),
        dx_ptr=dx,
        dx_stride=dx.stride(),
        dr_ptr=dr,
        dr_stride=None if dr is None else dr.stride(),
        dW_ptr=dW,
        dW_stride=None if dW is None else dW.stride(),
        s_ptr=s,
        s_stride=None if s is None else s.stride(),
        eps=eps,
        multiplier=multiplier,
        B=B,
        H=H,
        H_inv=1 / H,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
