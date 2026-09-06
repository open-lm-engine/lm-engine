# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2
from ....triton_utils import clamp, get_start_end, matmul, sigmoid, sigmoid_backward, tanh, tanh_backward
from ..utils import _get_num_heads
from .forward import _get_autotune_configs


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["BLOCK_SIZE_H"],
    reset_to_zero=["dx_ptr", "dxf_ptr", "dxr_ptr", "dW_ptr", "dWf_ptr", "dWr_ptr"],
)
@triton.jit
def _gru_backward_triton_kernel(
    x_ptr,
    x_stride,
    W_ptr,
    W_stride,
    z_ptr,
    z_stride,
    xf_ptr,
    xf_stride,
    Wf_ptr,
    Wf_stride,
    f_ptr,
    f_stride,
    xr_ptr,
    xr_stride,
    Wr_ptr,
    Wr_stride,
    r_ptr,
    r_stride,
    h0_ptr,
    h0_stride,
    y_ptr,
    y_stride,
    dx_ptr,
    dx_stride,
    dxf_ptr,
    dxf_stride,
    dxr_ptr,
    dxr_stride,
    dW_ptr,
    dW_stride,
    dWf_ptr,
    dWf_stride,
    dWr_ptr,
    dWr_stride,
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
    Gxf: tl.constexpr,
    Gxr: tl.constexpr,
    Gw: tl.constexpr,
    Gwf: tl.constexpr,
    Gwr: tl.constexpr,
    gradient_clipping,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(0)
    BLOCK_ID_N = tl.program_id(1)

    BLOCK_ID_Nx = BLOCK_ID_N // Gx
    BLOCK_ID_Nxf = BLOCK_ID_N // Gxf
    BLOCK_ID_Nxr = BLOCK_ID_N // Gxr

    BLOCK_ID_Nw = BLOCK_ID_N // Gw
    BLOCK_ID_Nwf = BLOCK_ID_N // Gwf
    BLOCK_ID_Nwr = BLOCK_ID_N // Gwr

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H

    MASK_BH = MASK_B[:, None] & MASK_H[None, :]
    MASK_HH = MASK_H[:, None] & MASK_H[None, :]

    if dht_ptr is None:
        dht = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W_ptr.dtype.element_ty)
    else:
        dht = tl.load(
            dht_ptr + BLOCK_B[:, None] * dht_stride[0] + BLOCK_ID_N * dht_stride[1] + BLOCK_H[None, :] * dht_stride[2],
            mask=MASK_BH,
        )

    dW = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
    dWf = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
    dWr = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)

    W = tl.load(
        W_ptr + BLOCK_ID_Nw * W_stride[0] + BLOCK_H[:, None] * W_stride[1] + BLOCK_H[None, :] * W_stride[2],
        mask=MASK_HH,
    )

    Wf = tl.load(
        Wf_ptr + BLOCK_ID_Nwf * Wf_stride[0] + BLOCK_H[:, None] * Wf_stride[1] + BLOCK_H[None, :] * Wf_stride[2],
        mask=MASK_HH,
    )

    Wr = tl.load(
        Wr_ptr + BLOCK_ID_Nwr * Wr_stride[0] + BLOCK_H[:, None] * Wr_stride[1] + BLOCK_H[None, :] * Wr_stride[2],
        mask=MASK_HH,
    )

    if h0_ptr is None:
        h0 = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W.dtype)
    else:
        h0 = tl.load(
            h0_ptr + BLOCK_B[:, None] * h0_stride[0] + BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] * h0_stride[2],
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

    if z_ptr is None:
        tl.static_assert(x_ptr is not None)
        x_ptrs = (
            x_ptr
            + BLOCK * x_stride[0]
            + S_LAST * x_stride[S_DIM]
            + BLOCK_ID_Nx * x_stride[N_DIM]
            + BLOCK_H[None, :] * x_stride[H_DIM]
        )
    else:
        z_ptrs = (
            z_ptr
            + BLOCK * z_stride[0]
            + S_LAST * z_stride[S_DIM]
            + BLOCK_ID_N * z_stride[N_DIM]
            + BLOCK_H[None, :] * z_stride[H_DIM]
        )

    if f_ptr is None:
        tl.static_assert(xf_ptr is not None)
        xf_ptrs = (
            xf_ptr
            + BLOCK * xf_stride[0]
            + S_LAST * xf_stride[S_DIM]
            + BLOCK_ID_Nxf * xf_stride[N_DIM]
            + BLOCK_H[None, :] * xf_stride[H_DIM]
        )
    else:
        f_ptrs = (
            f_ptr
            + BLOCK * f_stride[0]
            + S_LAST * f_stride[S_DIM]
            + BLOCK_ID_N * f_stride[N_DIM]
            + BLOCK_H[None, :] * f_stride[H_DIM]
        )

    if r_ptr is None:
        tl.static_assert(xr_ptr is not None)
        xr_ptrs = (
            xr_ptr
            + BLOCK * xr_stride[0]
            + S_LAST * xr_stride[S_DIM]
            + BLOCK_ID_Nxr * xr_stride[N_DIM]
            + BLOCK_H[None, :] * xr_stride[H_DIM]
        )
    else:
        r_ptrs = (
            r_ptr
            + BLOCK * r_stride[0]
            + S_LAST * r_stride[S_DIM]
            + BLOCK_ID_N * r_stride[N_DIM]
            + BLOCK_H[None, :] * r_stride[H_DIM]
        )

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

    dxf_ptrs = (
        dxf_ptr
        + BLOCK * dxf_stride[0]
        + S_LAST * dxf_stride[S_DIM]
        + BLOCK_ID_Nxf * dxf_stride[N_DIM]
        + BLOCK_H[None, :] * dxf_stride[H_DIM]
    )

    dxr_ptrs = (
        dxr_ptr
        + BLOCK * dxr_stride[0]
        + S_LAST * dxr_stride[S_DIM]
        + BLOCK_ID_Nxr * dxr_stride[N_DIM]
        + BLOCK_H[None, :] * dxr_stride[H_DIM]
    )

    dy_ptrs = (
        dy_ptr
        + BLOCK * dy_stride[0]
        + S_LAST * dy_stride[S_DIM]
        + BLOCK_ID_N * dy_stride[N_DIM]
        + BLOCK_H[None, :] * dy_stride[H_DIM]
    )

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        dh = dht
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        MASK = ((END >= START) & MASK_H[None, :]) if IS_VARLEN else MASK_BH
        y_ptrs -= y_stride[S_DIM]

        if IS_VARLEN:
            y_prev = tl.where(END > START, tl.load(y_ptrs, mask=MASK), h0)
        elif s == 0:
            y_prev = h0
        else:
            y_prev = tl.load(y_ptrs, mask=MASK)

        if r_ptr is None:
            x = tl.load(xr_ptrs, mask=MASK)
            xr_ptrs -= xr_stride[S_DIM]

            r = matmul(A=y_prev, B=Wr, C=x, output_dtype=tl.float32)
            r = sigmoid(r, output_dtype=x.dtype)
        else:
            r = tl.load(r_ptrs, mask=MASK)
            r_ptrs -= r_stride[S_DIM]

        if z_ptr is None:
            x = tl.load(x_ptrs, mask=MASK)
            x_ptrs -= x_stride[S_DIM]

            z = matmul(A=y_prev * r, B=W, C=x, output_dtype=tl.float32)
            z = tanh(z, output_dtype=x.dtype)
        else:
            z = tl.load(z_ptrs, mask=MASK)
            z_ptrs -= z_stride[S_DIM]

        if f_ptr is None:
            x = tl.load(xf_ptrs, mask=MASK)
            xf_ptrs -= xf_stride[S_DIM]

            f = matmul(A=y_prev, B=Wf, C=x, output_dtype=tl.float32)
            f = sigmoid(f, output_dtype=x.dtype)
        else:
            f = tl.load(f_ptrs, mask=MASK)
            f_ptrs -= f_stride[S_DIM]

        dy = tl.load(dy_ptrs, mask=MASK) + dh
        dy_ptrs -= dy_stride[S_DIM]

        dh = f * dy
        dz = dy * (1 - f)
        df = dy * (y_prev - z)

        dx = dz * tanh_backward(z)
        drh = matmul(A=dx, B=W.T, C=None, output_dtype=dx.dtype)

        if IS_VARLEN:
            y_prev = tl.where(MASK, y_prev, 0)

        dW = matmul(A=(r * y_prev).T, B=dx, C=dW, output_dtype=dW.dtype)

        if Gx == 1:
            tl.store(dx_ptrs, dx, mask=MASK)
        else:
            tl.atomic_add(dx_ptrs, dx, mask=MASK, sem="relaxed")

        dx_ptrs -= dx_stride[S_DIM]
        dh += drh * r

        dxf = df * sigmoid_backward(f)
        dh = matmul(A=dxf, B=Wf.T, C=dh, output_dtype=dx.dtype)
        dWf = matmul(A=y_prev.T, B=dxf, C=dWf, output_dtype=dW.dtype)

        if Gxf == 1:
            tl.store(dxf_ptrs, dxf, mask=MASK)
        else:
            tl.atomic_add(dxf_ptrs, dxf, mask=MASK, sem="relaxed")

        dxf_ptrs -= dxf_stride[S_DIM]

        dxr = drh * y_prev * sigmoid_backward(r)
        dh = matmul(A=dxr, B=Wr.T, C=dh, output_dtype=dx.dtype)
        dWr = matmul(A=y_prev.T, B=dxr, C=dWr, output_dtype=dW.dtype)

        dht = tl.where(MASK, dh, dht) if IS_VARLEN else dh

        if Gxr == 1:
            tl.store(dxr_ptrs, dxr, mask=MASK)
        else:
            tl.atomic_add(dxr_ptrs, dxr, mask=MASK, sem="relaxed")

        dxr_ptrs -= dxr_stride[S_DIM]

        if IS_VARLEN:
            END -= 1

    if dh0_ptr is not None:
        tl.store(
            dh0_ptr + BLOCK_B[:, None] * dh0_stride[0] + BLOCK_ID_N * dh0_stride[1] + BLOCK_H[None, :] * dh0_stride[2],
            dht,
            mask=MASK_BH,
        )

    tl.atomic_add(
        dW_ptr + BLOCK_ID_Nw * dW_stride[0] + BLOCK_H[:, None] * dW_stride[1] + BLOCK_H[None, :] * dW_stride[2],
        dW,
        mask=MASK_HH,
        sem="relaxed",
    )

    tl.atomic_add(
        dWf_ptr + BLOCK_ID_Nwf * dWf_stride[0] + BLOCK_H[:, None] * dWf_stride[1] + BLOCK_H[None, :] * dWf_stride[2],
        dWf,
        mask=MASK_HH,
        sem="relaxed",
    )

    tl.atomic_add(
        dWr_ptr + BLOCK_ID_Nwr * dWr_stride[0] + BLOCK_H[:, None] * dWr_stride[1] + BLOCK_H[None, :] * dWr_stride[2],
        dWr,
        mask=MASK_HH,
        sem="relaxed",
    )


@xma_op(mutates_args={"dxf", "dWf", "dxr", "dWr", "dx", "dW", "dh0"})
def _gru_backward_triton(
    x: torch.Tensor | None,
    W: torch.Tensor,
    y: torch.Tensor,
    xf: torch.Tensor | None,
    Wf: torch.Tensor,
    f: torch.Tensor | None,
    dxf: torch.Tensor,
    dWf: torch.Tensor,
    xr: torch.Tensor | None,
    Wr: torch.Tensor,
    r: torch.Tensor | None,
    dxr: torch.Tensor,
    dWr: torch.Tensor,
    z: torch.Tensor | None,
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
        assert max_seqlen is None
        B, S, _, H = y.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = max_seqlen
        H = y.size(-1)

    Nx, Nxf, Nxr, Nw, Nwf, Nwr, N = _get_num_heads(x=dx, W=W, xf=dxf, Wf=Wf, xr=dxr, Wr=Wr, run_check=False)

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda kwargs: (ceil_divide(B, kwargs["BLOCK_SIZE_B"]), N)

    _gru_backward_triton_kernel[GRID](
        x_ptr=x,
        x_stride=None if x is None else x.stride(),
        W_ptr=W,
        W_stride=W.stride(),
        z_ptr=z,
        z_stride=None if z is None else z.stride(),
        xf_ptr=xf,
        xf_stride=None if xf is None else xf.stride(),
        Wf_ptr=Wf,
        Wf_stride=Wf.stride(),
        f_ptr=f,
        f_stride=None if f is None else f.stride(),
        xr_ptr=xr,
        xr_stride=None if xr is None else xr.stride(),
        Wr_ptr=Wr,
        Wr_stride=Wr.stride(),
        r_ptr=r,
        r_stride=None if r is None else r.stride(),
        h0_ptr=h0,
        h0_stride=None if h0 is None else h0.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        dx_ptr=dx,
        dx_stride=dx.stride(),
        dxf_ptr=dxf,
        dxf_stride=dxf.stride(),
        dxr_ptr=dxr,
        dxr_stride=dxr.stride(),
        dW_ptr=dW,
        dW_stride=dW.stride(),
        dWf_ptr=dWf,
        dWf_stride=dWf.stride(),
        dWr_ptr=dWr,
        dWr_stride=dWr.stride(),
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
        Gxf=N // Nxf,
        Gxr=N // Nxr,
        Gw=N // Nw,
        Gwf=N // Nwf,
        Gwr=N // Nwr,
        gradient_clipping=gradient_clipping,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
