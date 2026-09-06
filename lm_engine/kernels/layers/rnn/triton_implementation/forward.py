# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2, get_powers_of_2
from ....triton_utils import get_start_end, matmul, tanh


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(4, 8):
        for num_stages in range(1, 5):
            for BLOCK_SIZE_B in [1] + get_powers_of_2(16, 32):
                configs.append(
                    triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B}, num_stages=num_stages, num_warps=num_warps)
                )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"])
@triton.jit
def _rnn_forward_triton_kernel(
    x_ptr,
    x_stride,
    W_ptr,
    W_stride,
    h0_ptr,
    h0_stride,
    y_ptr,
    y_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    B,
    S,
    H: tl.constexpr,
    Gx: tl.constexpr,
    Gw: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(0)
    BLOCK_ID_N = tl.program_id(1)

    BLOCK_ID_Nx = BLOCK_ID_N // Gx
    BLOCK_ID_Nw = BLOCK_ID_N // Gw

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H

    MASK_BH = MASK_B[:, None] & MASK_H[None, :]
    MASK_HH = MASK_H[:, None] & MASK_H[None, :]

    W = tl.load(
        W_ptr + BLOCK_ID_Nw * W_stride[0] + BLOCK_H[:, None] * W_stride[1] + BLOCK_H[None, :] * W_stride[2],
        mask=MASK_HH,
    )

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(
            h0_ptr + BLOCK_B[:, None] * h0_stride[0] + BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] * h0_stride[2],
            mask=MASK_BH,
        )

    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None
    S_DIM: tl.constexpr = 1 - IS_VARLEN
    N_DIM: tl.constexpr = 2 - IS_VARLEN
    H_DIM: tl.constexpr = 3 - IS_VARLEN

    if IS_VARLEN:
        START, END = get_start_end(cu_seqlens_ptr, cu_seqlens_stride, BLOCK_B, MASK_B)

    BLOCK = START if IS_VARLEN else BLOCK_B[:, None]
    x_ptrs = x_ptr + BLOCK * x_stride[0] + BLOCK_ID_Nx * x_stride[N_DIM] + BLOCK_H[None, :] * x_stride[H_DIM]
    y_ptrs = y_ptr + BLOCK * y_stride[0] + BLOCK_ID_N * y_stride[N_DIM] + BLOCK_H[None, :] * y_stride[H_DIM]

    for _ in range(S):
        MASK = ((START < END) & MASK_H[None, :]) if IS_VARLEN else MASK_BH

        x = tl.load(x_ptrs, mask=MASK)
        x_ptrs += x_stride[S_DIM]

        h = matmul(A=h, B=W, C=x, output_dtype=tl.float32)
        h = tanh(h, output_dtype=x.dtype)

        tl.store(y_ptrs, h, mask=MASK)
        y_ptrs += y_stride[S_DIM]

        if IS_VARLEN:
            START += 1


@xma_op(mutates_args={"y"})
def _rnn_forward_triton(
    x: torch.Tensor,
    W: torch.Tensor,
    h0: torch.Tensor | None,
    y: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    if cu_seqlens is None:
        B, S, Nx, H = x.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = max_seqlen
        _, Nx, H = x.size()

    Nw = W.size(0)
    N = max(Nx, Nw)

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda kwargs: (ceil_divide(B, kwargs["BLOCK_SIZE_B"]), N)

    _rnn_forward_triton_kernel[GRID](
        x_ptr=x,
        x_stride=x.stride(),
        W_ptr=W,
        W_stride=W.stride(),
        h0_ptr=h0,
        h0_stride=None if h0 is None else h0.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        cu_seqlens_ptr=cu_seqlens,
        cu_seqlens_stride=None if cu_seqlens is None else cu_seqlens.stride(),
        B=B,
        S=S,
        H=H,
        Gx=N // Nx,
        Gw=N // Nw,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
