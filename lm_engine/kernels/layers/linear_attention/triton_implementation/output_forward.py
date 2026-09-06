# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import triton
import triton.language as tl

from ....math import get_powers_of_2
from ....triton_utils import matmul


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(4, 8):
        for num_stages in range(1, 5):
            for BLOCK_SIZE_K in get_powers_of_2(32, 64):
                for BLOCK_SIZE_V in get_powers_of_2(32, 64):
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                                "BLOCK_SIZE_V": BLOCK_SIZE_V,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=[])
@triton.jit
def _output_forward_triton_kernel(
    q_ptr,
    q_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    h0_ptr,
    h0_stride,
    h_ptr,
    h_stride,
    y_ptr,
    y_stride,
    attention_multiplier,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    S,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    Gq: tl.constexpr,
    Gk: tl.constexpr,
    Gv: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    BLOCK_ID_BN = tl.program_id(0)
    BLOCK_ID_S = tl.program_id(1)
    BLOCK_ID_V = tl.program_id(2)

    BLOCK_ID_B = BLOCK_ID_BN // N
    BLOCK_ID_N = BLOCK_ID_BN % N

    BLOCK_ID_Nq = BLOCK_ID_N // Gq
    BLOCK_ID_Nk = BLOCK_ID_N // Gk
    BLOCK_ID_Nv = BLOCK_ID_N // Gv

    BLOCK_S = BLOCK_ID_S * BLOCK_SIZE_S + tl.arange(0, BLOCK_SIZE_S)
    BLOCK_K = tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = BLOCK_ID_V * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)

    MASK_S = BLOCK_S < S
    MASK_V = BLOCK_V < V

    MASK_SV = MASK_S[:, None] & MASK_V[None, :]
    CAUSAL_MASK = (BLOCK_S[:, None] >= BLOCK_S[None, :]) & MASK_S[:, None] & MASK_S[None, :]

    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None
    S_DIM: tl.constexpr = 1 - IS_VARLEN
    N_DIM: tl.constexpr = 2 - IS_VARLEN
    K_DIM: tl.constexpr = 3 - IS_VARLEN

    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_ID_B * cu_seqlens_stride[0]
        start = tl.load(cu_seqlens_ptrs)
        end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0])

        S = end - start
        BLOCK = start + BLOCK_S

    _B = BLOCK[:, None] if IS_VARLEN else BLOCK_ID_B
    _S = 0 if IS_VARLEN else BLOCK_S[:, None]

    q_ptrs = (
        q_ptr
        + _B * q_stride[0]
        + _S * q_stride[S_DIM]
        + BLOCK_ID_Nq * q_stride[N_DIM]
        + BLOCK_K[None, :] * q_stride[K_DIM]
    )

    k_ptrs = (
        k_ptr
        + _B * k_stride[0]
        + _S * k_stride[S_DIM]
        + BLOCK_ID_Nk * k_stride[N_DIM]
        + BLOCK_K[None, :] * k_stride[K_DIM]
    )

    v = tl.load(
        v_ptr
        + _B * v_stride[0]
        + _S * v_stride[S_DIM]
        + BLOCK_ID_Nv * v_stride[N_DIM]
        + BLOCK_V[None, :] * v_stride[K_DIM],
        mask=MASK_SV,
    )

    qk = tl.zeros((BLOCK_SIZE_S, BLOCK_SIZE_S), dtype=tl.float32)
    y = tl.zeros((BLOCK_SIZE_S, BLOCK_SIZE_V), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
        MASK_K = BLOCK_K < K

        MASK_SK = MASK_S[:, None] & MASK_K[None, :]
        MASK_KV = MASK_K[:, None] & MASK_V[None, :]

        q = tl.load(q_ptrs, mask=MASK_SK)
        q_ptrs += BLOCK_SIZE_S * q_stride[S_DIM]

        k = tl.load(k_ptrs, mask=MASK_SK)
        k_ptrs += BLOCK_SIZE_S * k_stride[S_DIM]

        if BLOCK_ID_S == 0:
            if h0_ptr is None:
                h = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=q.dtype)
            else:
                h = tl.load(
                    h0_ptr
                    + BLOCK_ID_B * h0_stride[0]
                    + BLOCK_ID_N * h0_stride[1]
                    + BLOCK_K[:, None] * h0_stride[2]
                    + BLOCK_V[None, :] * h0_stride[3],
                    mask=MASK_KV,
                )
        else:
            h = tl.load(
                h_ptr
                + BLOCK_ID_B * h_stride[0]
                + (BLOCK_ID_S - 1) * h_stride[1]
                + BLOCK_ID_N * h_stride[2]
                + BLOCK_K[:, None] * h_stride[3]
                + BLOCK_V[None, :] * h_stride[4],
                mask=MASK_KV,
            )

        y = matmul(A=q, B=h, C=y, output_dtype=y.dtype)
        qk = matmul(A=q, B=k.T, C=qk, output_dtype=qk.dtype)

        BLOCK_K += BLOCK_SIZE_K

    qk = tl.where(CAUSAL_MASK, qk, 0)

    y = matmul(A=qk.to(v.dtype), B=v, C=y, output_dtype=tl.float32)
    y *= attention_multiplier

    tl.store(
        y_ptr
        + _B * y_stride[0]
        + _S * y_stride[S_DIM]
        + BLOCK_ID_N * y_stride[N_DIM]
        + BLOCK_V[None, :] * y_stride[K_DIM],
        y,
        mask=MASK_SV,
    )
