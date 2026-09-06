# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2
from ....triton_utils import get_start_end, matmul, sigmoid, tanh
from ...rnn.triton_implementation.forward import _get_autotune_configs
from ..utils import _get_num_heads


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"])
@triton.jit
def _gru_forward_triton_kernel(
    x_ptr,
    x_stride,
    xf_ptr,
    xf_stride,
    xr_ptr,
    xr_stride,
    W_ptr,
    W_stride,
    Wf_ptr,
    Wf_stride,
    Wr_ptr,
    Wr_stride,
    z_ptr,
    z_stride,
    f_ptr,
    f_stride,
    r_ptr,
    r_stride,
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
    Gxf: tl.constexpr,
    Gxr: tl.constexpr,
    Gw: tl.constexpr,
    Gwf: tl.constexpr,
    Gwr: tl.constexpr,
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
    xf_ptrs = xf_ptr + BLOCK * xf_stride[0] + BLOCK_ID_Nxf * xf_stride[N_DIM] + BLOCK_H[None, :] * xf_stride[H_DIM]
    xr_ptrs = xr_ptr + BLOCK * xr_stride[0] + BLOCK_ID_Nxr * xr_stride[N_DIM] + BLOCK_H[None, :] * xr_stride[H_DIM]
    y_ptrs = y_ptr + BLOCK * y_stride[0] + BLOCK_ID_N * y_stride[N_DIM] + BLOCK_H[None, :] * y_stride[H_DIM]

    if z_ptr is not None:
        z_ptrs = z_ptr + BLOCK * z_stride[0] + BLOCK_ID_N * z_stride[N_DIM] + BLOCK_H[None, :] * z_stride[H_DIM]

    if r_ptr is not None:
        r_ptrs = r_ptr + BLOCK * r_stride[0] + BLOCK_ID_N * r_stride[N_DIM] + BLOCK_H[None, :] * r_stride[H_DIM]

    if f_ptr is not None:
        f_ptrs = f_ptr + BLOCK * f_stride[0] + BLOCK_ID_N * f_stride[N_DIM] + BLOCK_H[None, :] * f_stride[H_DIM]

    for _ in range(S):
        MASK = ((START < END) & MASK_H[None, :]) if IS_VARLEN else MASK_BH

        x = tl.load(xr_ptrs, mask=MASK)
        xr_ptrs += xr_stride[S_DIM]

        r = matmul(A=h, B=Wr, C=x, output_dtype=tl.float32)
        r = sigmoid(r, output_dtype=x.dtype)

        if r_ptr is not None:
            tl.store(r_ptrs, r, mask=MASK)
            r_ptrs += r_stride[S_DIM]

        x = tl.load(x_ptrs, mask=MASK)
        x_ptrs += x_stride[S_DIM]

        z = matmul(A=h * r, B=W, C=x, output_dtype=tl.float32)
        z = tanh(z, output_dtype=x.dtype)

        if z_ptr is not None:
            tl.store(z_ptrs, z, mask=MASK)
            z_ptrs += z_stride[S_DIM]

        x = tl.load(xf_ptrs, mask=MASK)
        xf_ptrs += xf_stride[S_DIM]

        f = matmul(A=h, B=Wf, C=x, output_dtype=tl.float32)
        f = sigmoid(f, output_dtype=x.dtype)

        if f_ptr is not None:
            tl.store(f_ptrs, f, mask=MASK)
            f_ptrs += f_stride[S_DIM]

        h = f * h + (1 - f) * z

        tl.store(y_ptrs, h, mask=MASK)
        y_ptrs += y_stride[S_DIM]

        if IS_VARLEN:
            START += 1


@xma_op(mutates_args={"f", "r", "z", "y"})
def _gru_forward_triton(
    x: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    Wf: torch.Tensor,
    f: torch.Tensor | None,
    xr: torch.Tensor,
    Wr: torch.Tensor,
    r: torch.Tensor | None,
    z: torch.Tensor | None,
    h0: torch.Tensor | None,
    y: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    if cu_seqlens is None:
        assert max_seqlen is None
        B, S, _, H = x.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = max_seqlen
        H = x.size(-1)

    Nx, Nxf, Nxr, Nw, Nwf, Nwr, N = _get_num_heads(x=x, W=W, xf=xf, Wf=Wf, xr=xr, Wr=Wr, run_check=False)

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda kwargs: (ceil_divide(B, kwargs["BLOCK_SIZE_B"]), N)

    _gru_forward_triton_kernel[GRID](
        x_ptr=x,
        x_stride=x.stride(),
        xf_ptr=xf,
        xf_stride=xf.stride(),
        xr_ptr=xr,
        xr_stride=xr.stride(),
        W_ptr=W,
        W_stride=W.stride(),
        Wf_ptr=Wf,
        Wf_stride=Wf.stride(),
        Wr_ptr=Wr,
        Wr_stride=Wr.stride(),
        z_ptr=z,
        z_stride=None if z is None else z.stride(),
        f_ptr=f,
        f_stride=None if f is None else f.stride(),
        r_ptr=r,
        r_stride=None if r is None else r.stride(),
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
        Gxf=N // Nxf,
        Gxr=N // Nxr,
        Gw=N // Nw,
        Gwf=N // Nwf,
        Gwr=N // Nwr,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
