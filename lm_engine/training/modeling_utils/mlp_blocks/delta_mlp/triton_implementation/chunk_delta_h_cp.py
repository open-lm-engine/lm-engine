# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from einops import repeat
from fla.ops.cp.comm import all_gather_into_tensor
from fla.utils import USE_CUDA_GRAPH, autotune_cache_kwargs, check_shared_mem


if TYPE_CHECKING:
    from fla.ops.cp.context import FLACPContext


# Distribute M over (I - k^T w): M - (k^T w) @ M. Algebraically identical, but the identity never
# enters the dot, so its tf32 truncation scales with ||k^T w|| instead of ||I||.
_DISTRIBUTE_M = True


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_H0": lambda args: args["h0"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=[
        "H",
        "HV",
        "K",
        "V",
        "BT",
        "BLOCK_SIZE",
        "BK1",
        "IS_VARLEN",
        "MULTI_SEQS",
        "HAS_H0",
        "FUSE_U",
        "DISTRIBUTE_M",
    ],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def pre_process_fwd_kernel_merged(
    k,
    v,
    w,
    u,
    beta,
    A,
    hm,
    h0,
    cu_seqlens,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MULTI_SEQS: tl.constexpr,
    HAS_H0: tl.constexpr,
    FUSE_U: tl.constexpr,
    DISTRIBUTE_M: tl.constexpr,
):
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    i_hv = i_h // (H // HV)
    if MULTI_SEQS:
        i_n = tl.program_id(2)
        # Offset hm for this subseq: hm[i_n, h, k, v+k]
        hm += i_n * H * K * (K + V) + i_h * K * (K + V)
    else:
        i_n = 0
        hm += i_h * K * (K + V)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # Determine if this block handles h (V part) or m (K part)
    # i_col is in range [0, cdiv(V + K, BLOCK_SIZE))
    # Columns [0, V) are for h, columns [V, V+K) are for m
    is_h_part = i_col * BLOCK_SIZE < V
    # For DPLR (USE_BG), w and bg share the same head dim H as k/ag.
    # For GDN/KDA, w has head dim HV (same as v).
    k += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    stride_k = H * K
    stride_w = H * K

    if is_h_part:
        # ====== Stage 1: Compute h (K x V) ======
        if FUSE_U:
            v += ((bos * HV + i_hv) * V).to(tl.int64)
            beta += (bos * H + i_h).to(tl.int64)
            A += ((bos * H + i_h) * BT).to(tl.int64)
        else:
            u += ((bos * H + i_h) * V).to(tl.int64)

        stride_v = HV * V
        stride_u = H * V
        i_v = i_col

        # Initialize h accumulators
        if HAS_H0:
            p_h0_h1 = tl.make_block_ptr(
                h0 + i_h * K * V, (K, V), (V, 1), (0, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0)
            )
            b_h1 = tl.load(p_h0_h1, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_h1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            if HAS_H0:
                p_h0_h2 = tl.make_block_ptr(
                    h0 + i_h * K * V, (K, V), (V, 1), (64, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0)
                )
                b_h2 = tl.load(p_h0_h2, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_h2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            if HAS_H0:
                p_h0_h3 = tl.make_block_ptr(
                    h0 + i_h * K * V, (K, V), (V, 1), (128, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0)
                )
                b_h3 = tl.load(p_h0_h3, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_h3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            if HAS_H0:
                p_h0_h4 = tl.make_block_ptr(
                    h0 + i_h * K * V, (K, V), (V, 1), (192, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0)
                )
                b_h4 = tl.load(p_h0_h4, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_h4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        # Main recurrence for h
        for i_t in range(NT):
            # Compute decayed v
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_decay = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v_decay += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v_decay += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v_decay += tl.dot(b_w, b_h4.to(b_w.dtype))

            if FUSE_U:
                p_v = tl.make_block_ptr(
                    v, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0)
                )
                b_v = tl.load(p_v, boundary_check=(0, 1))

                p_b = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
                b_b = tl.load(p_b, boundary_check=(0,))

                p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
                b_A = tl.load(p_A, boundary_check=(0, 1))

                b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
                b_u = tl.dot(b_A.to(b_vb.dtype), b_vb, allow_tf32=False) - b_v_decay

            else:
                p_u = tl.make_block_ptr(
                    u, (T, V), (stride_u, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0)
                )

                # GDN/KDA mode: v_new = v - w @ h
                b_u = tl.load(p_u, boundary_check=(0, 1)) - b_v_decay

            b_u = b_u.to(k.dtype.element_ty)

            # Update h
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))

            # GDN/KDA mode: h += k^T @ v_new
            b_h1 += tl.dot(b_k, b_u)
            if K > 64:
                p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h2 += tl.dot(b_k, b_u)
            if K > 128:
                p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h3 += tl.dot(b_k, b_u)
            if K > 192:
                p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h4 += tl.dot(b_k, b_u)

        # Store h results
        stride_hm_kv = K + V
        p_h1 = tl.make_block_ptr(hm, (K, V), (stride_hm_kv, 1), (0, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(hm, (K, V), (stride_hm_kv, 1), (64, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(hm, (K, V), (stride_hm_kv, 1), (128, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(hm, (K, V), (stride_hm_kv, 1), (192, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))
    else:
        # ====== Stage 2: Compute m (K x K) ======
        # i_col is for m part, map to K dimension
        # m starts at column V, so offset = i_col * BLOCK_SIZE - V
        # Use tl.cdiv to correctly compute the number of blocks for V dimension
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        # Following stage2 kernel design:
        # - BK1 is the full K dimension (next_power_of_2(K))
        # - BLOCK_SIZE is the column block size (like BK2=32 in stage2)
        # Each block computes a (BK1, BLOCK_SIZE) sub-matrix of m
        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        # Initialize as identity matrix: M_0 = I
        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for i_t in range(NT):
            # Load k and w with full BK1 rows
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

            # Compute m update
            # GDN/KDA mode: M = (diag - k^T @ w) @ M
            b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)

            if DISTRIBUTE_M:
                b_m = b_m - tl.dot(b_kw.to(tl.float32), b_m.to(tl.float32))
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)
                b_m_i = b_diag - b_kw
                b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32))

        # Store m result
        stride_hm_kv = K + V
        p_m = tl.make_block_ptr(
            hm + V, (K, K), (stride_hm_kv, 1), (0, i_k_col * BLOCK_SIZE), (BK1, BLOCK_SIZE), (1, 0)
        )
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "HAS_H0": lambda args: args["h0"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=["H", "K", "V", "BK", "FORWARD", "INTRACARD_MODE", "HAS_H0", "TRANSPOSE_STATE"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["pre_or_post_num_ranks", "rank", "NUM_SEQ_ENTRIES"])
def merge_fwd_bwd_kernel(
    h,  # [H, K, V] or [num_non_first, H, K, V] for intracard (or [V, K] when transposed)
    ag_hm,  # [H, K, K+V] or [S_split, H, K, K+V] for intracard (always [K, V+K])
    pre_or_post_num_ranks,  # num_ranks for CP, NUM_SPLIT_SEQS for intracard
    rank,  # rank for CP, not used for intracard
    seq_offsets,  # None for CP, [num_split_seqs+1] for intracard
    init_offsets,  # None for CP, [num_split_seqs+1] for intracard
    h0_seq_ids,  # None for CP, [num_split_seqs] for intracard
    h0,  # None or [N_orig, H, K, V] for intracard (or [V, K] when transposed)
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    FORWARD: tl.constexpr,  # True for FWD, False for BWD
    INTRACARD_MODE: tl.constexpr,  # True: intracard mode, False: CP mode
    NUM_SEQ_ENTRIES,  # num_split_seqs for intracard
    HAS_H0: tl.constexpr,  # Heuristic: whether h0 is provided
    TRANSPOSE_STATE: tl.constexpr = False,  # When True, h0/h use [V, K] layout; ag_hm always [K, V+K]
):
    """
    Unified merge kernel for both CP and Intra-card modes.

    CP mode (INTRACARD_MODE=False):
        Grid: (V/BV, H)
        Merges across ranks for context parallel.

    Intra-card mode (INTRACARD_MODE=True):
        Grid: (V/BV, NUM_SEQ_ENTRIES, H)
        Merges across subseqs within card for intra-card context parallel.

    When TRANSPOSE_STATE=True, h0 and output h use [V, K] layout.
    ag_hm always uses [K, V+K] layout (from pre_scan).
    The recurrence h' = M @ h + he becomes h_T' = h_T @ M^T + he^T.
    """
    i_v = tl.program_id(0)
    if INTRACARD_MODE:
        i_seq = tl.program_id(1)
        i_h = tl.program_id(2)

        if i_seq >= NUM_SEQ_ENTRIES:
            return

        # Load offsets for this sequence
        ss_start = tl.load(seq_offsets + i_seq).to(tl.int64)
        ss_end = tl.load(seq_offsets + i_seq + 1).to(tl.int64)
        init_base = tl.load(init_offsets + i_seq).to(tl.int64)
        num_subseqs = ss_end - ss_start

        stride_hm_s = H * K * (V + K)
        stride_hm_h = K * (V + K)

        # Initialize from h0 if provided
        if HAS_H0:
            orig_seq_id = tl.load(h0_seq_ids + i_seq).to(tl.int64)
            if TRANSPOSE_STATE:
                p_h0 = tl.make_block_ptr(
                    h0 + (orig_seq_id * H + i_h) * V * K, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0)
                )
                b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
            else:
                p_h0 = tl.make_block_ptr(
                    h0 + (orig_seq_id * H + i_h) * K * V, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)
                )
                b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        else:
            if TRANSPOSE_STATE:
                b_h = tl.zeros([BV, BK], dtype=tl.float32)
            else:
                b_h = tl.zeros([BK, BV], dtype=tl.float32)

        # Merge loop over subseqs
        for idx in range(num_subseqs):
            i_ss = ss_start + idx
            base = i_ss * stride_hm_s + i_h * stride_hm_h

            # he and m are always in [K, V+K] layout from pre_scan
            p_he = tl.make_block_ptr(ag_hm + base, (K, V), (V + K, 1), (0, i_v * BV), (BK, BV), (1, 0))
            b_he = tl.load(p_he, boundary_check=(0, 1)).to(tl.float32)
            p_m = tl.make_block_ptr(ag_hm + base + V, (K, K), (V + K, 1), (0, 0), (BK, BK), (1, 0))
            b_m = tl.load(p_m, boundary_check=(0, 1)).to(tl.float32)
            if TRANSPOSE_STATE:
                # h_T' = h_T @ M^T + he^T
                b_h = tl.dot(b_h.to(tl.float32), tl.trans(b_m)) + tl.trans(b_he)
            else:
                b_h = tl.dot(b_m.to(tl.float32), b_h.to(tl.float32)) + b_he.to(tl.float32)

            # Store for non-first subseqs
            if idx < num_subseqs - 1:
                init_idx = init_base + idx
                stride_init = H * K * V
                if TRANSPOSE_STATE:
                    p_out = tl.make_block_ptr(
                        h + init_idx * stride_init + i_h * V * K, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0)
                    )
                else:
                    p_out = tl.make_block_ptr(
                        h + init_idx * stride_init + i_h * K * V, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)
                    )
                tl.store(p_out, b_h.to(p_out.dtype.element_ty), boundary_check=(0, 1))
    else:
        # CP mode
        i_h = tl.program_id(1)
        num_ranks = pre_or_post_num_ranks.to(tl.int64)
        h += i_h * K * V
        ag_hm += i_h * K * (K + V)
        stride = H * K * (K + V)

        # Initialize from h0 if provided
        if HAS_H0:
            if TRANSPOSE_STATE:
                p_h0 = tl.make_block_ptr(h0 + i_h * V * K, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
                b_h0 = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
            else:
                p_h0 = tl.make_block_ptr(h0 + i_h * K * V, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
                b_h0 = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        else:
            if TRANSPOSE_STATE:
                b_h0 = tl.zeros([BV, BK], dtype=tl.float32)
            else:
                b_h0 = tl.zeros([BK, BV], dtype=tl.float32)

        b_h = b_h0
        for idx in range(num_ranks):
            if FORWARD:
                cur_rank = rank - num_ranks + idx
            else:
                cur_rank = rank + num_ranks - idx
            p_ag_h = tl.make_block_ptr(ag_hm + cur_rank * stride, (K, V), (K + V, 1), (0, i_v * BV), (BK, BV), (1, 0))
            b_ag_h = tl.load(p_ag_h, boundary_check=(0, 1))
            p_ag_m = tl.make_block_ptr(ag_hm + cur_rank * stride + V, (K, K), (K + V, 1), (0, 0), (BK, BK), (1, 0))
            b_ag_m = tl.load(p_ag_m, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h = tl.dot(
                    (b_h.to(tl.float32) - b_h0.to(tl.float32)).to(tl.float32), tl.trans(b_ag_m).to(tl.float32)
                ) + tl.trans(b_ag_h).to(tl.float32)
            else:
                b_h = tl.dot(
                    b_ag_m.to(tl.float32), (b_h.to(tl.float32) - b_h0.to(tl.float32)).to(tl.float32)
                ) + b_ag_h.to(tl.float32)
        if TRANSPOSE_STATE:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_DHT": lambda args: args["dht"] is not None,
        "FUSE_DV_LOCAL": lambda args: args["A_local"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": block_size}, num_warps=num_warps, num_stages=num_stages)
        for block_size in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in ([4, 3, 2] if check_shared_mem("ampere") else [1])
    ],
    key=["H", "HO", "K", "V", "BT", "BK1", "IS_VARLEN", "HAS_DHT", "FUSE_DV_LOCAL", "DISTRIBUTE_M"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def pre_process_bwd_kernel_merged(
    q,
    k,
    w,
    do,
    dhm,
    dv,
    A_local,
    dht,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HO: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_DHT: tl.constexpr,
    FUSE_DV_LOCAL: tl.constexpr,
    DISTRIBUTE_M: tl.constexpr,
):
    """
    Merged backward kernel that computes both dh (K x V) and dm (K x K) in a single kernel.

    Similar to pre_process_fwd_kernel_merged, this kernel uses a unified grid where:
    - Columns [0, V) are for computing dh (stage 1)
    - Columns [V, V+K) are for computing dm (stage 2)
    """
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # Determine if this block handles dh (V part) or dm (K part)
    is_dh_part = i_col * BLOCK_SIZE < V

    # Calculate offsets
    # For DPLR (USE_BG), w shares the same head dim H as q/k/ag.
    # For GDN/KDA, w has head dim HV (same as do/dv).
    q += ((bos * H + i_h) * K).to(tl.int64)
    k += ((bos * H + i_h) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    dhm += i_h * K * (V + K)
    stride_qk = H * K
    stride_w = H * K

    if is_dh_part:
        # ====== Stage 1: Compute dh (K x V) ======
        i_ho = i_h // (H // HO)
        do += ((bos * HO + i_ho) * V).to(tl.int64)
        if FUSE_DV_LOCAL:
            A_local += ((bos * H + i_h) * BT).to(tl.int64)
        else:
            dv += ((bos * H + i_h) * V).to(tl.int64)
        stride_do = HO * V
        stride_v = H * V
        i_v = i_col

        # Initialize dh accumulators
        if HAS_DHT:
            p_dht1 = tl.make_block_ptr(
                dht + i_h * K * V, (K, V), (V, 1), (0, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0)
            )
            b_dh1 = tl.load(p_dht1, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_dh1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            if HAS_DHT:
                p_dht2 = tl.make_block_ptr(
                    dht + i_h * K * V, (K, V), (V, 1), (64, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0)
                )
                b_dh2 = tl.load(p_dht2, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_dh2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            if HAS_DHT:
                p_dht3 = tl.make_block_ptr(
                    dht + i_h * K * V, (K, V), (V, 1), (128, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0)
                )
                b_dh3 = tl.load(p_dht3, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_dh3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            if HAS_DHT:
                p_dht4 = tl.make_block_ptr(
                    dht + i_h * K * V, (K, V), (V, 1), (192, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0)
                )
                b_dh4 = tl.load(p_dht4, boundary_check=(0, 1)).to(tl.float32)
            else:
                b_dh4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        # Main recurrence for dh (reverse order)
        for i_t in range(NT - 1, -1, -1):

            if not FUSE_DV_LOCAL:
                p_dv = tl.make_block_ptr(
                    dv, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0)
                )
            p_do = tl.make_block_ptr(
                do, (T, V), (stride_do, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0)
            )
            b_do = tl.load(p_do, boundary_check=(0, 1))

            # Update dv
            p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

            if K > 64:
                p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

            if K > 128:
                p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

            if K > 192:
                p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

            if FUSE_DV_LOCAL:
                # dv_local is rebuilt in-kernel from A_local
                p_A_local = tl.make_block_ptr(A_local, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
                b_A_local = tl.load(p_A_local, boundary_check=(0, 1))
                b_dv += tl.dot(b_A_local, b_do)
            else:
                b_dv += tl.load(p_dv, boundary_check=(0, 1))

            # Update dh
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_w), (0, i_t * BT), (64, BT), (0, 1))
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (0, i_t * BT), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 64:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (64, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_w), (64, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 128:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (128, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_w), (128, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 192:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (192, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_w), (192, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

        # Store dh results
        p_dh1 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (0, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (64, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (128, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dhm, (K, V), (V + K, 1), (192, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))
    else:
        # ====== Stage 2: Compute dm (K x K) ======
        # i_col is for dm part, map to K dimension
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        # Following stage2 kernel design for backward (FORWARD=False)
        # - BK1 is the full K dimension (next_power_of_2(K))
        # - BLOCK_SIZE is the column block size
        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        # Initialize as identity matrix: M_0 = I
        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for _i_t in range(NT):
            # Reverse order for backward
            i_t = NT - 1 - _i_t

            # Load k and w with full BK1 rows
            p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

            # Compute dm update for backward
            b_kw = tl.dot(tl.trans(b_w), b_k.to(b_w.dtype))

            if DISTRIBUTE_M:
                b_m = b_m - tl.dot(b_kw.to(tl.float32), b_m.to(tl.float32))
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)
                # GDN/KDA mode: dM = (diag - w^T @ k) @ dM
                b_m_i = b_diag - b_kw
                # Keep m chain in fp32 to avoid precision loss from repeated bf16 casting
                b_m = tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32))

        # Store dm result
        p_m = tl.make_block_ptr(dhm + V, (K, K), (V + K, 1), (0, i_k_col * BLOCK_SIZE), (BK1, BLOCK_SIZE), (1, 0))
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h_pre_process(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    A: torch.Tensor | None = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    cu_seqlens: torch.LongTensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
    fuse_u: bool = False,
) -> torch.Tensor:
    if context is None or context.group is None:
        return initial_state
    rank = dist.get_rank(group=context.group)

    if fuse_u:
        assert beta is not None
        assert A is not None
        assert u is None
    else:
        assert beta is None
        assert A is None
        assert u is not None

    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = chunk_size
    BK = triton.next_power_of_2(K)

    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    assert H >= HV and H % HV == 0

    hm = k.new_zeros(H, K, (V + K), dtype=torch.float32)
    if transpose_state_layout:
        initial_state_cp = k.new_empty(N, H, V, K, dtype=torch.float32)
    else:
        initial_state_cp = k.new_empty(N, H, K, V, dtype=torch.float32)

    if initial_state is not None:
        assert not transpose_state_layout
        assert initial_state.ndim == 3
        assert initial_state.shape == initial_state_cp.shape[1:]
        initial_state_cp.copy_(repeat(initial_state, "... -> n ...", n=N))
    else:
        initial_state_cp.zero_()

    if not context.is_last_rank:
        BLOCK_SIZE = 32 if K <= 64 else 64
        grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), H)
        pre_process_fwd_kernel_merged[grid](
            k=k,
            v=v,
            w=w,
            u=u,
            beta=beta,
            A=A,
            hm=hm,
            h0=initial_state,
            cu_seqlens=cu_seqlens[-2:],
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            BLOCK_SIZE=BLOCK_SIZE,
            MULTI_SEQS=False,
            FUSE_U=fuse_u,
            DISTRIBUTE_M=_DISTRIBUTE_M,
        )
    ag_hm, _ = all_gather_into_tensor(hm, group=context.group)
    if not context.is_first_rank:

        def grid(meta):
            return (triton.cdiv(V, meta["BV"]), H)

        merge_fwd_bwd_kernel[grid](
            h=initial_state_cp[0],
            ag_hm=ag_hm,
            pre_or_post_num_ranks=context.pre_num_ranks,
            rank=rank,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=initial_state,
            H=H,
            K=K,
            V=V,
            BK=BK,
            FORWARD=True,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            TRANSPOSE_STATE=transpose_state_layout,
        )
    return initial_state_cp


def chunk_gated_delta_rule_bwd_dhu_pre_process(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor | None,
    A_local: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    dht: torch.Tensor | None = None,
    context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
) -> torch.Tensor | None:
    if context is None or context.group is None:
        return dht
    rank = dist.get_rank(context.group)

    B, T, H, K, V, HO = *q.shape, do.shape[-1], do.shape[2]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert K <= 256, "current kernel does not support head dimension being larger than 256."
    assert H % HO == 0, f"H={H} must be a multiple of HO={HO}"
    BK = triton.next_power_of_2(K)
    if A_local is None:
        assert dv is not None
    else:
        # dv_local is rebuilt in-kernel from A_local
        assert dv is None
        assert A_local.shape == (B, T, H, BT)
        assert A_local.dtype == do.dtype == q.dtype

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1

    dhm = q.new_zeros(H, K, V + K, dtype=torch.float32)
    if transpose_state_layout:
        dht_cp = q.new_empty(N, H, V, K, dtype=torch.float32)
    else:
        dht_cp = q.new_empty(N, H, K, V, dtype=torch.float32)

    if dht is not None:
        assert not transpose_state_layout
        assert dht.ndim == 3
        assert dht.shape == dht_cp.shape[1:]
        dht_cp.copy_(repeat(dht, "... -> n ...", n=N))
    else:
        dht_cp.zero_()

    if not context.is_first_rank:

        def grid(meta):
            return (triton.cdiv(V, meta["BLOCK_SIZE"]) + triton.cdiv(K, meta["BLOCK_SIZE"]), H)

        pre_process_bwd_kernel_merged[grid](
            q=q,
            k=k,
            w=w,
            do=do,
            dhm=dhm,
            dv=dv,
            A_local=A_local,
            dht=dht,
            cu_seqlens=cu_seqlens[:2],
            scale=scale,
            T=T,
            H=H,
            HO=HO,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            DISTRIBUTE_M=_DISTRIBUTE_M,
        )

    ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)

    if not context.is_last_rank:

        def grid(meta):
            return (triton.cdiv(V, meta["BV"]), H)

        merge_fwd_bwd_kernel[grid](
            h=dht_cp[-1],
            ag_hm=ag_dhm,
            pre_or_post_num_ranks=context.post_num_ranks,
            rank=rank,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=dht,
            H=H,
            K=K,
            V=V,
            BK=BK,
            FORWARD=False,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            TRANSPOSE_STATE=transpose_state_layout,
        )

    return dht_cp
