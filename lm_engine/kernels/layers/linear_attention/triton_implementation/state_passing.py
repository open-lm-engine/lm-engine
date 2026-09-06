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
            for BLOCK_SIZE_S in [1] + get_powers_of_2(16, 64):
                for BLOCK_SIZE_K in get_powers_of_2(32, 64):
                    for BLOCK_SIZE_V in get_powers_of_2(32, 64):
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_SIZE_S": BLOCK_SIZE_S,
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
def _state_passing_forward_triton_kernel(
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
    ht_ptr,
    ht_stride,
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
    CHUNK_SIZE: tl.constexpr,
):
    tl.static_assert(CHUNK_SIZE % BLOCK_SIZE_S == 0)

    if q_ptr is not None:
        tl.static_assert(y_ptr is not None)
    else:
        tl.static_assert(y_ptr is None)

    BLOCK_ID_BN = tl.program_id(0)
    BLOCK_ID_K = tl.program_id(1)
    BLOCK_ID_V = tl.program_id(2)

    BLOCK_ID_B = BLOCK_ID_BN // N
    BLOCK_ID_N = BLOCK_ID_BN % N

    BLOCK_ID_Nq = BLOCK_ID_N // Gq
    BLOCK_ID_Nk = BLOCK_ID_N // Gk
    BLOCK_ID_Nv = BLOCK_ID_N // Gv

    BLOCK_S = tl.arange(0, BLOCK_SIZE_S)
    BLOCK_K = BLOCK_ID_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = BLOCK_ID_V * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)

    MASK_K = BLOCK_K < K
    MASK_V = BLOCK_V < V

    MASK_KV = MASK_K[:, None] & MASK_V[None, :]
    CAUSAL_MASK = BLOCK_S[:, None] >= BLOCK_S[None, :]

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=tl.float32)
    else:
        h = tl.load(
            h0_ptr
            + BLOCK_ID_B * h0_stride[0]
            + BLOCK_ID_N * h0_stride[1]
            + BLOCK_K[:, None] * h0_stride[2]
            + BLOCK_V[None, :] * h0_stride[3],
            mask=MASK_KV,
        ).to(tl.float32)

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

    if q_ptr is not None:
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

    v_ptrs = (
        v_ptr
        + _B * v_stride[0]
        + _S * v_stride[S_DIM]
        + BLOCK_ID_Nv * v_stride[N_DIM]
        + BLOCK_V[None, :] * v_stride[K_DIM]
    )

    if y_ptr is not None:
        y_ptrs = (
            y_ptr
            + _B * y_stride[0]
            + _S * y_stride[S_DIM]
            + BLOCK_ID_N * y_stride[N_DIM]
            + BLOCK_V[None, :] * y_stride[K_DIM]
        )

    if h_ptr is not None:
        h_ptrs = (
            h_ptr
            + _B * h_stride[0]
            + BLOCK_ID_N * h_stride[2]
            + BLOCK_K[:, None] * h_stride[3]
            + BLOCK_V[None, :] * h_stride[4]
        )

    NUM_BLOCKS_S = tl.cdiv(S, BLOCK_SIZE_S)

    for s in range(1, NUM_BLOCKS_S + 1):
        MASK_S = BLOCK_S < S

        MASK_SK = MASK_S[:, None] & MASK_K[None, :]
        MASK_SV = MASK_S[:, None] & MASK_V[None, :]

        k = tl.load(k_ptrs, mask=MASK_SK)
        k_ptrs += BLOCK_SIZE_S * k_stride[S_DIM]

        v = tl.load(v_ptrs, mask=MASK_SV)
        v_ptrs += BLOCK_SIZE_S * v_stride[S_DIM]

        if q_ptr is not None:
            q = tl.load(q_ptrs, mask=MASK_SK)
            q_ptrs += BLOCK_SIZE_S * q_stride[S_DIM]

            qk = matmul(A=q, B=k.T, C=None, output_dtype=q.dtype)
            qk = tl.where(CAUSAL_MASK, qk, 0)

            y = matmul(A=qk, B=v, C=None, output_dtype=tl.float32)
            y = matmul(A=q, B=h.to(q.dtype), C=y, output_dtype=tl.float32)
            y *= attention_multiplier

            tl.store(y_ptrs, y, mask=MASK_SV)
            y_ptrs += BLOCK_SIZE_S * y_stride[S_DIM]

        h = matmul(A=k.T, B=v, C=h, output_dtype=h.dtype)

        if h_ptr is not None and (s * BLOCK_SIZE_S) % CHUNK_SIZE == 0 and s != NUM_BLOCKS_S:
            tl.store(h_ptrs, h, mask=MASK_KV)
            h_ptrs += h_stride[S_DIM]

        BLOCK_S += BLOCK_SIZE_S

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
