# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2, get_powers_of_2
from ....triton_utils import matmul, tanh


_MAX_BLOCK_SIZE_K = 64


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(1, 32):
        for num_stages in range(1, 5):
            configs.append(triton.Config({}, num_stages=num_stages, num_warps=num_warps))

    return configs


@triton.jit
def _forward_single_step(h_prev, W, k, v, f):
    x = matmul(A=k[:, None], B=v[None, :], C=None, output_dtype=k.dtype)
    z = matmul(A=h_prev, B=W, C=x, output_dtype=tl.float32)
    z = tanh(z, output_dtype=x.dtype)

    h = f * h_prev + (1 - f) * z

    return z, h


@triton.jit
def _forward(
    q_ptr,
    q_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    W_ptr,
    W_stride,
    xf_ptr,
    xf_stride,
    h0_ptr,
    h0_stride,
    h_ptr,
    h_stride,
    ht_ptr,
    ht_stride,
    y_ptr,
    y_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
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

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=k_ptr.dtype.element_ty)
    else:
        h = tl.load(
            h0_ptr
            + BLOCK_ID_B * h0_stride[0]
            + BLOCK_ID_N * h0_stride[1]
            + BLOCK_K[:, None] * h0_stride[2]
            + BLOCK_V[None, :] * h0_stride[3],
            mask=MASK_KV,
        )

    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None
    S_DIM: tl.constexpr = 1 - IS_VARLEN
    N_DIM: tl.constexpr = 2 - IS_VARLEN
    K_DIM: tl.constexpr = 3 - IS_VARLEN

    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_ID_B * cu_seqlens_stride[0]
        START = tl.load(cu_seqlens_ptrs)
        END = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0])

        S = END - START

    _B = START if IS_VARLEN else BLOCK_ID_B

    k_ptrs = k_ptr + _B * k_stride[0] + BLOCK_ID_Nk * k_stride[N_DIM] + BLOCK_K * k_stride[K_DIM]
    v_ptrs = v_ptr + _B * v_stride[0] + BLOCK_ID_Nv * v_stride[N_DIM] + BLOCK_V * v_stride[K_DIM]
    xf_ptrs = xf_ptr + _B * xf_stride[0] + BLOCK_ID_Nxf * xf_stride[N_DIM]

    if h_ptr is not None:
        h_ptrs = (
            h_ptr
            + tl.cast(_B * h_stride[0], tl.uint32)
            + tl.cast(BLOCK_ID_N * h_stride[N_DIM], tl.uint32)
            + tl.cast(BLOCK_K[:, None] * h_stride[K_DIM], tl.uint32)
            + tl.cast(BLOCK_V[None, :] * h_stride[K_DIM + 1], tl.uint32)
        )

    if q_ptr is not None:
        q_ptrs = q_ptr + _B * q_stride[0] + BLOCK_ID_Nq * q_stride[N_DIM] + BLOCK_K * q_stride[K_DIM]

    if y_ptr is not None:
        y_ptrs = y_ptr + _B * y_stride[0] + BLOCK_ID_N * y_stride[N_DIM] + BLOCK_V * y_stride[K_DIM]

    for s in range(1, S + 1):
        k = tl.load(k_ptrs, mask=MASK_K)
        k_ptrs += k_stride[S_DIM]

        v = tl.load(v_ptrs, mask=MASK_V)
        v_ptrs += v_stride[S_DIM]

        f = tl.load(xf_ptrs)
        xf_ptrs += xf_stride[S_DIM]

        _, h = _forward_single_step(h_prev=h, W=W, k=k, v=v, f=f)

        # only cache on start of each chunk or at the end of the full RNN
        if h_ptr is not None:
            tl.store(h_ptrs, h, mask=MASK_KV)
            h_ptrs += h_stride[S_DIM]

        if y_ptr is not None:
            q = tl.load(q_ptrs, mask=MASK_K)
            q_ptrs += q_stride[S_DIM]

            y = matmul(A=q[None, :], B=h, C=None, output_dtype=q.dtype)

            if ATOMIC_ADD:
                tl.atomic_add(y_ptrs[None, :], y, mask=MASK_V[None, :])
            else:
                tl.store(y_ptrs[None, :], y, mask=MASK_V[None, :])

            y_ptrs += y_stride[S_DIM]

    if ht_ptr is not None:
        tl.store(
            ht_ptr
            + BLOCK_ID_B * ht_stride[0]
            + BLOCK_ID_N * ht_stride[1]
            + BLOCK_K[:, None] * ht_stride[2]
            + BLOCK_V[None, :] * ht_stride[3],
            h,
            mask=MASK_KV,
        )


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_K", "BLOCK_SIZE_V"], reset_to_zero=["y_ptr"])
@triton.jit
def _m2rnn_forward_triton_kernel(
    q_ptr,
    q_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    W_ptr,
    W_stride,
    xf_ptr,
    xf_stride,
    h0_ptr,
    h0_stride,
    h_ptr,
    h_stride,
    ht_ptr,
    ht_stride,
    y_ptr,
    y_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
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
    _forward(
        q_ptr=q_ptr,
        q_stride=q_stride,
        k_ptr=k_ptr,
        k_stride=k_stride,
        v_ptr=v_ptr,
        v_stride=v_stride,
        W_ptr=W_ptr,
        W_stride=W_stride,
        xf_ptr=xf_ptr,
        xf_stride=xf_stride,
        h0_ptr=h0_ptr,
        h0_stride=h0_stride,
        h_ptr=h_ptr,
        h_stride=h_stride,
        ht_ptr=ht_ptr,
        ht_stride=ht_stride,
        y_ptr=y_ptr,
        y_stride=y_stride,
        cu_seqlens_ptr=cu_seqlens_ptr,
        cu_seqlens_stride=cu_seqlens_stride,
        S=S,
        K=K,
        V=V,
        Gq=Gq,
        Gk=Gk,
        Gv=Gv,
        Gw=Gw,
        Gxf=Gxf,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        ATOMIC_ADD=ATOMIC_ADD,
    )


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_K", "BLOCK_SIZE_V"])
@triton.jit
def _m2rnn_forward_no_output_triton_kernel(
    q_ptr,
    q_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    W_ptr,
    W_stride,
    xf_ptr,
    xf_stride,
    h0_ptr,
    h0_stride,
    h_ptr,
    h_stride,
    ht_ptr,
    ht_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
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
):
    _forward(
        q_ptr=q_ptr,
        q_stride=q_stride,
        k_ptr=k_ptr,
        k_stride=k_stride,
        v_ptr=v_ptr,
        v_stride=v_stride,
        W_ptr=W_ptr,
        W_stride=W_stride,
        xf_ptr=xf_ptr,
        xf_stride=xf_stride,
        h0_ptr=h0_ptr,
        h0_stride=h0_stride,
        h_ptr=h_ptr,
        h_stride=h_stride,
        ht_ptr=ht_ptr,
        ht_stride=ht_stride,
        y_ptr=None,
        y_stride=None,
        cu_seqlens_ptr=cu_seqlens_ptr,
        cu_seqlens_stride=cu_seqlens_stride,
        S=S,
        K=K,
        V=V,
        Gq=Gq,
        Gk=Gk,
        Gv=Gv,
        Gw=Gw,
        Gxf=Gxf,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        ATOMIC_ADD=None,
    )


@xma_op(mutates_args={"h", "ht", "y"})
def _m2rnn_forward_triton(
    q: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    h0: torch.Tensor | None,
    h: torch.Tensor | None,
    ht: torch.Tensor | None,
    y: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    Nq: int,
    Nk: int,
    Nv: int,
    Nw: int,
    Nxf: int,
    N: int,
) -> None:
    if cu_seqlens is None:
        B, S, _, K = k.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        K = k.size(-1)

    V = v.size(-1)

    BLOCK_SIZE_K = get_next_power_of_2(K)
    BLOCK_SIZE_K = max(16, BLOCK_SIZE_K)
    BLOCK_SIZE_K = min(_MAX_BLOCK_SIZE_K, BLOCK_SIZE_K)
    NUM_BLOCKS_K = ceil_divide(K, BLOCK_SIZE_K)

    BLOCK_SIZE_V = get_next_power_of_2(V)
    BLOCK_SIZE_V = max(16, BLOCK_SIZE_V)

    kwargs = dict(
        q_ptr=q,
        q_stride=None if q is None else q.stride(),
        k_ptr=k,
        k_stride=k.stride(),
        v_ptr=v,
        v_stride=v.stride(),
        W_ptr=W,
        W_stride=W.stride(),
        xf_ptr=xf,
        xf_stride=xf.stride(),
        h0_ptr=h0,
        h0_stride=None if h0 is None else h0.stride(),
        h_ptr=h,
        h_stride=None if h is None else h.stride(),
        ht_ptr=ht,
        ht_stride=None if ht is None else ht.stride(),
        cu_seqlens_ptr=cu_seqlens,
        cu_seqlens_stride=None if cu_seqlens is None else cu_seqlens.stride(),
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
    )

    if y is None:
        _m2rnn_forward_no_output_triton_kernel[B, N, NUM_BLOCKS_K](**kwargs)
    else:
        _m2rnn_forward_triton_kernel[B, N, NUM_BLOCKS_K](
            **kwargs, y_ptr=y, y_stride=y.stride(), ATOMIC_ADD=K > BLOCK_SIZE_K
        )
