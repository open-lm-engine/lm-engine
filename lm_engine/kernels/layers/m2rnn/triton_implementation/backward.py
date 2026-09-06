# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2
from ....triton_utils import clamp, matmul, tanh_backward
from ..utils import _get_num_heads
from .forward import _MAX_BLOCK_SIZE_K, _forward_single_step, _get_autotune_configs


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["BLOCK_SIZE_K", "BLOCK_SIZE_V"],
    reset_to_zero=["dxf_ptr", "dq_ptr", "dk_ptr", "dv_ptr", "dW_ptr"],
)
@triton.jit
def _m2rnn_backward_triton_kernel(
    q_ptr,
    q_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    W_ptr,
    W_stride,
    h_ptr,
    h_stride,
    xf_ptr,
    xf_stride,
    dxf_ptr,
    dxf_stride,
    h0_ptr,
    h0_stride,
    dy_ptr,
    dy_stride,
    dht_ptr,
    dht_stride,
    dq_ptr,
    dq_stride,
    dk_ptr,
    dk_stride,
    dv_ptr,
    dv_stride,
    dW_ptr,
    dW_stride,
    dh0_ptr,
    dh0_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    gradient_clipping,
    S,
    K: tl.constexpr,
    V: tl.constexpr,
    Gq: tl.constexpr,
    Gk: tl.constexpr,
    Gv: tl.constexpr,
    Gw: tl.constexpr,
    Gxf: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(0)
    BLOCK_ID_N = tl.program_id(1)
    BLOCK_ID_K = tl.program_id(2)

    BLOCK_ID_Nq = BLOCK_ID_N // Gq
    BLOCK_ID_Nk = BLOCK_ID_N // Gk
    BLOCK_ID_Nv = BLOCK_ID_N // Gv

    BLOCK_ID_Nw = BLOCK_ID_N // Gw
    BLOCK_ID_Nxf = BLOCK_ID_N // Gxf

    BLOCK_K = BLOCK_ID_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)

    MASK_K = BLOCK_K < K
    MASK_V = BLOCK_V < V

    MASK_KV = MASK_K[:, None] & MASK_V[None, :]
    MASK_VV = MASK_V[:, None] & MASK_V[None, :]

    W = tl.load(
        W_ptr + BLOCK_ID_Nw * W_stride[0] + BLOCK_V[:, None] * W_stride[1] + BLOCK_V[None, :] * W_stride[2],
        mask=MASK_VV,
    )

    if dht_ptr is None:
        dh = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=W_ptr.dtype.element_ty)
    else:
        dh = tl.load(
            dht_ptr
            + BLOCK_ID_B * dht_stride[0]
            + BLOCK_ID_N * dht_stride[1]
            + BLOCK_K[:, None] * dht_stride[2]
            + BLOCK_V[None, :] * dht_stride[3],
            mask=MASK_KV,
        )

    dW = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_V), dtype=tl.float32)

    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None
    S_DIM: tl.constexpr = 1 - IS_VARLEN
    N_DIM: tl.constexpr = 2 - IS_VARLEN
    K_DIM: tl.constexpr = 3 - IS_VARLEN

    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_ID_B * cu_seqlens_stride[0]
        START = tl.load(cu_seqlens_ptrs)
        END = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0])

        S = END - START

    _B = END - 1 if IS_VARLEN else BLOCK_ID_B
    _S = 0 if IS_VARLEN else S - 1

    q_ptrs = (
        q_ptr + _B * q_stride[0] + _S * q_stride[S_DIM] + BLOCK_ID_Nq * q_stride[N_DIM] + BLOCK_K * q_stride[K_DIM]
    )

    k_ptrs = (
        k_ptr + _B * k_stride[0] + _S * k_stride[S_DIM] + BLOCK_ID_Nk * k_stride[N_DIM] + BLOCK_K * k_stride[K_DIM]
    )

    v_ptrs = (
        v_ptr + _B * v_stride[0] + _S * v_stride[S_DIM] + BLOCK_ID_Nv * v_stride[N_DIM] + BLOCK_V * v_stride[K_DIM]
    )

    dy_ptrs = (
        dy_ptr + _B * dy_stride[0] + _S * dy_stride[S_DIM] + BLOCK_ID_N * dy_stride[N_DIM] + BLOCK_V * dy_stride[K_DIM]
    )

    dq_ptrs = (
        dq_ptr
        + _B * dq_stride[0]
        + _S * dq_stride[S_DIM]
        + BLOCK_ID_Nq * dq_stride[N_DIM]
        + BLOCK_K * dq_stride[K_DIM]
    )

    dk_ptrs = (
        dk_ptr
        + _B * dk_stride[0]
        + _S * dk_stride[S_DIM]
        + BLOCK_ID_Nk * dk_stride[N_DIM]
        + BLOCK_K * dk_stride[K_DIM]
    )

    dv_ptrs = (
        dv_ptr
        + _B * dv_stride[0]
        + _S * dv_stride[S_DIM]
        + BLOCK_ID_Nv * dv_stride[N_DIM]
        + BLOCK_V * dv_stride[K_DIM]
    )

    xf_ptrs = xf_ptr + _B * xf_stride[0] + _S * xf_stride[S_DIM] + BLOCK_ID_Nxf * xf_stride[N_DIM]

    h_ptrs = (
        h_ptr
        + tl.cast(_B * h_stride[0], tl.uint32)
        + tl.cast(_S * h_stride[S_DIM], tl.uint32)
        + tl.cast(BLOCK_ID_N * h_stride[N_DIM], tl.uint32)
        + tl.cast(BLOCK_K[:, None] * h_stride[K_DIM], tl.uint32)
        + tl.cast(BLOCK_V[None, :] * h_stride[K_DIM + 1], tl.uint32)
    )

    dxf_ptrs = dxf_ptr + _B * dxf_stride[0] + _S * dxf_stride[S_DIM] + BLOCK_ID_Nxf * dxf_stride[N_DIM]

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if s == 0:
            if h0_ptr is None:
                h_prev = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=W.dtype)
            else:
                h_prev = tl.load(
                    h0_ptr
                    + BLOCK_ID_B * h0_stride[0]
                    + BLOCK_ID_N * h0_stride[1]
                    + BLOCK_K[:, None] * h0_stride[2]
                    + BLOCK_V[None, :] * h0_stride[3],
                    mask=MASK_KV,
                )
        else:
            h_ptrs -= h_stride[S_DIM]
            h_prev = tl.load(h_ptrs, mask=MASK_KV)

        q = tl.load(q_ptrs, mask=MASK_K)
        q_ptrs -= q_stride[S_DIM]

        k = tl.load(k_ptrs, mask=MASK_K)
        k_ptrs -= k_stride[S_DIM]

        v = tl.load(v_ptrs, mask=MASK_V)
        v_ptrs -= v_stride[S_DIM]

        f = tl.load(xf_ptrs)
        xf_ptrs -= xf_stride[S_DIM]

        z, h = _forward_single_step(h_prev=h_prev, W=W, k=k, v=v, f=f)

        dy = tl.load(dy_ptrs, mask=MASK_V)
        dy_ptrs -= dy_stride[S_DIM]

        dq = matmul(A=h, B=dy[:, None], C=None, output_dtype=q.dtype)

        if Gq == 1:
            tl.store(dq_ptrs[:, None], dq, mask=MASK_K[:, None])
        else:
            tl.atomic_add(dq_ptrs[:, None], dq, mask=MASK_K[:, None], sem="relaxed")

        dq_ptrs -= dq_stride[S_DIM]

        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        dyh = matmul(A=q[:, None], B=dy[None, :], C=dh, output_dtype=q.dtype)

        df = dyh * (h_prev - z)
        df = tl.sum(df)

        if Gxf == 1 and not ATOMIC_ADD:
            tl.store(dxf_ptrs, df)
        else:
            tl.atomic_add(dxf_ptrs, df, sem="relaxed")

        dxf_ptrs -= dxf_stride[S_DIM]

        dh = f * dyh
        dz = dyh * (1 - f)

        dx = dz * tanh_backward(z)
        dh = matmul(A=dx, B=W.T, C=dh, output_dtype=dx.dtype)
        dW = matmul(A=h_prev.T, B=dx, C=dW, output_dtype=dW.dtype)

        dv = matmul(A=dx.T, B=k[:, None], C=None, output_dtype=k.dtype)

        if Gv == 1 and not ATOMIC_ADD:
            tl.store(dv_ptrs[:, None], dv, mask=MASK_V[:, None])
        else:
            tl.atomic_add(dv_ptrs[:, None], dv, mask=MASK_V[:, None], sem="relaxed")

        dv_ptrs -= dv_stride[S_DIM]
        dk = matmul(A=dx, B=v[:, None], C=None, output_dtype=k.dtype)

        if Gk == 1:
            tl.store(dk_ptrs[:, None], dk, mask=MASK_K[:, None])
        else:
            tl.atomic_add(dk_ptrs[:, None], dk, mask=MASK_K[:, None], sem="relaxed")

        dk_ptrs -= dk_stride[S_DIM]

    if dh0_ptr is not None:
        tl.store(
            dh0_ptr
            + BLOCK_ID_B * dh0_stride[0]
            + BLOCK_ID_N * dh0_stride[1]
            + BLOCK_K[:, None] * dh0_stride[2]
            + BLOCK_V[None, :] * dh0_stride[3],
            dh,
            mask=MASK_KV,
        )

    tl.atomic_add(
        dW_ptr + BLOCK_ID_Nw * dW_stride[0] + BLOCK_V[:, None] * dW_stride[1] + BLOCK_V[None, :] * dW_stride[2],
        dW,
        mask=MASK_VV,
        sem="relaxed",
    )


@xma_op(mutates_args={"dq", "dk", "dv", "dW", "dxf", "dh0"})
def _m2rnn_backward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    h0: torch.Tensor | None,
    dy: torch.Tensor,
    dht: torch.Tensor | None,
    h: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dW: torch.Tensor,
    dxf: torch.Tensor,
    dh0: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    gradient_clipping: float | None,
) -> None:
    Nq, Nk, Nv, Nw, Nxf, N = _get_num_heads(q=q, k=k, v=v, W=W, xf=xf, run_check=False)

    if cu_seqlens is None:
        B, S, _, K, V = h.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        _, _, K, V = h.size()

    BLOCK_SIZE_K = get_next_power_of_2(K)
    BLOCK_SIZE_K = max(16, BLOCK_SIZE_K)
    BLOCK_SIZE_K = min(_MAX_BLOCK_SIZE_K, BLOCK_SIZE_K)

    BLOCK_SIZE_V = get_next_power_of_2(V)
    BLOCK_SIZE_V = max(16, BLOCK_SIZE_V)

    _m2rnn_backward_triton_kernel[B, N, ceil_divide(K, BLOCK_SIZE_K)](
        q_ptr=q,
        q_stride=q.stride(),
        k_ptr=k,
        k_stride=k.stride(),
        v_ptr=v,
        v_stride=v.stride(),
        W_ptr=W,
        W_stride=W.stride(),
        h_ptr=h,
        h_stride=h.stride(),
        xf_ptr=xf,
        xf_stride=xf.stride(),
        dxf_ptr=dxf,
        dxf_stride=dxf.stride(),
        h0_ptr=h0,
        h0_stride=None if h0 is None else h0.stride(),
        dy_ptr=dy,
        dy_stride=dy.stride(),
        dht_ptr=dht,
        dht_stride=None if dht is None else dht.stride(),
        dq_ptr=dq,
        dq_stride=dq.stride(),
        dk_ptr=dk,
        dk_stride=dk.stride(),
        dv_ptr=dv,
        dv_stride=dv.stride(),
        dW_ptr=dW,
        dW_stride=dW.stride(),
        dh0_ptr=dh0,
        dh0_stride=None if dh0 is None else dh0.stride(),
        cu_seqlens_ptr=cu_seqlens,
        cu_seqlens_stride=None if cu_seqlens is None else cu_seqlens.stride(),
        gradient_clipping=gradient_clipping,
        S=S,
        K=K,
        V=V,
        Gq=N // Nq,
        Gk=N // Nk,
        Gv=N // Nv,
        Gw=N // Nw,
        Gxf=N // Nxf,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        ATOMIC_ADD=K > BLOCK_SIZE_K,
    )
