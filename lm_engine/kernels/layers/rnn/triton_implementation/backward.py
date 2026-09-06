# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2
from ....triton_utils import clamp, get_start_end, matmul, tanh_backward
from .forward import _get_autotune_configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"], reset_to_zero=["dx_ptr", "dW_ptr"])
@triton.jit
def _rnn_backward_triton_kernel(
    W_ptr,
    W_stride,
    h0_ptr,
    h0_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    dW_ptr,
    dW_stride,
    dh0_ptr,
    dh0_stride,
    dy_ptr,
    dy_stride,
    dht_ptr,
    dht_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    B,
    S,
    H: tl.constexpr,
    Gx: tl.constexpr,
    Gw: tl.constexpr,
    gradient_clipping,
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

    dW = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)

    if h0_ptr is None:
        h0 = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W.dtype)
    else:
        h0 = tl.load(
            h0_ptr + BLOCK_B[:, None] * h0_stride[0] + BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] * h0_stride[2],
            mask=MASK_BH,
        )

    if dht_ptr is None:
        dh = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W_ptr.dtype.element_ty)
    else:
        dh = tl.load(
            dht_ptr + BLOCK_B[:, None] * dht_stride[0] + BLOCK_ID_N * dht_stride[1] + BLOCK_H[None, :] * dht_stride[2],
            mask=MASK_BH,
        )

    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None
    S_DIM: tl.constexpr = 1 - IS_VARLEN
    N_DIM: tl.constexpr = 2 - IS_VARLEN
    H_DIM: tl.constexpr = 3 - IS_VARLEN

    if IS_VARLEN:
        START, END = get_start_end(cu_seqlens_ptr, cu_seqlens_stride, BLOCK_B, MASK_B)
        END -= 1

    BLOCK = END if IS_VARLEN else BLOCK_B[:, None]
    S_LAST = 0 if IS_VARLEN else S - 1

    y_ptrs = (
        y_ptr
        + BLOCK * y_stride[0]
        + S_LAST * y_stride[S_DIM]
        + BLOCK_ID_N * y_stride[N_DIM]
        + BLOCK_H[None, :] * y_stride[H_DIM]
    )

    dx_ptrs = (
        dx_ptr
        + BLOCK * dx_stride[0]
        + S_LAST * dx_stride[S_DIM]
        + BLOCK_ID_Nx * dx_stride[N_DIM]
        + BLOCK_H[None, :] * dx_stride[H_DIM]
    )

    dy_ptrs = (
        dy_ptr
        + BLOCK * dy_stride[0]
        + S_LAST * dy_stride[S_DIM]
        + BLOCK_ID_N * dy_stride[N_DIM]
        + BLOCK_H[None, :] * dy_stride[H_DIM]
    )

    y = tl.load(y_ptrs, mask=((END >= START) & MASK_H[None, :]) if IS_VARLEN else MASK_BH)
    y_ptrs -= y_stride[S_DIM]

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        MASK = ((END >= START) & MASK_H[None, :]) if IS_VARLEN else MASK_BH

        if IS_VARLEN:
            y_prev = tl.where(END > START, tl.load(y_ptrs, mask=MASK), h0)
        elif s == 0:
            y_prev = h0
        else:
            y_prev = tl.load(y_ptrs, mask=MASK)

        y_ptrs -= y_stride[S_DIM]

        dy = tl.load(dy_ptrs, mask=MASK) + dh
        dy_ptrs -= dy_stride[S_DIM]

        dx = dy * tanh_backward(y)

        _dh = matmul(A=dx, B=W.T, C=None, output_dtype=dx.dtype)
        dh = tl.where(MASK, _dh, dh) if IS_VARLEN else _dh

        if IS_VARLEN:
            y_prev = tl.where(MASK, y_prev, 0)

        dW = matmul(A=y_prev.T, B=dx, C=dW, output_dtype=dW.dtype)
        y = y_prev

        if Gx == 1:
            tl.store(dx_ptrs, dx, mask=MASK)
        else:
            tl.atomic_add(dx_ptrs, dx, mask=MASK, sem="relaxed")

        dx_ptrs -= dx_stride[S_DIM]

        if IS_VARLEN:
            END -= 1

    if dh0_ptr is not None:
        tl.store(
            dh0_ptr + BLOCK_B[:, None] * dh0_stride[0] + BLOCK_ID_N * dh0_stride[1] + BLOCK_H[None, :] * dh0_stride[2],
            dh,
            mask=MASK_BH,
        )

    tl.atomic_add(
        dW_ptr + BLOCK_ID_Nw * dW_stride[0] + BLOCK_H[:, None] * dW_stride[1] + BLOCK_H[None, :] * dW_stride[2],
        dW,
        mask=MASK_HH,
        sem="relaxed",
    )


@xma_op(mutates_args={"dx", "dW", "dh0"})
def _rnn_backward_triton(
    W: torch.Tensor,
    y: torch.Tensor,
    h0: torch.Tensor | None,
    dy: torch.Tensor,
    dht: torch.Tensor | None,
    dx: torch.Tensor,
    dW: torch.Tensor,
    dh0: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
    gradient_clipping: float | None,
) -> None:
    if cu_seqlens is None:
        B, S, N, H = y.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = max_seqlen
        _, N, H = y.size()

    Nx = dx.size(-2)
    Nw = W.size(0)

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)

    GRID = lambda kwargs: (ceil_divide(B, kwargs["BLOCK_SIZE_B"]), N)

    _rnn_backward_triton_kernel[GRID](
        W_ptr=W,
        W_stride=W.stride(),
        h0_ptr=h0,
        h0_stride=None if h0 is None else h0.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        dx_ptr=dx,
        dx_stride=dx.stride(),
        dW_ptr=dW,
        dW_stride=dW.stride(),
        dh0_ptr=dh0,
        dh0_stride=None if dh0 is None else dh0.stride(),
        dy_ptr=dy,
        dy_stride=dy.stride(),
        dht_ptr=dht,
        dht_stride=None if dht is None else dht.stride(),
        cu_seqlens_ptr=cu_seqlens,
        cu_seqlens_stride=None if cu_seqlens is None else cu_seqlens.stride(),
        B=B,
        S=S,
        H=H,
        Gx=N // Nx,
        Gw=N // Nw,
        gradient_clipping=gradient_clipping,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
