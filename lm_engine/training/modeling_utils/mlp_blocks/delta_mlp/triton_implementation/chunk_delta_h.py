# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from fla.ops.backends import dispatch
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp, exp2
from fla.utils import IS_NVIDIA_HOPPER, USE_CUDA_GRAPH, autotune_cache_kwargs, check_shared_mem

from .chunk_o import NUM_WARPS


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV, "SKIP_STORE": skip_store}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([2, 3, 4] if check_shared_mem("ampere") else [2, 1])
        for BV in ([32, 64] if check_shared_mem("ada") else [32])
        for skip_store in [True, False]
    ],
    key=[
        "H",
        "HV",
        "K",
        "V",
        "BT",
        "USE_INITIAL_STATE",
        "STORE_FINAL_STATE",
        "SAVE_NEW_VALUE",
        "TRANSPOSE_STATE",
        "IS_VARLEN",
        "FUSE_O",
        "CKPT_PAIR",
    ],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    q,
    k,
    v,
    beta,
    A,
    w,
    v_new,
    o,
    h,
    h0,
    ht,
    scale,
    cu_seqlens,
    chunk_offsets,
    T,
    NT_CKPT,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    FUSE_O: tl.constexpr,
    CKPT_PAIR: tl.constexpr,
    SKIP_STORE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    i_hv = i_h // (H // HV)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

        if not CKPT_PAIR:
            boh = i_n * NT
        else:
            # one dummy slot
            boh = i_n * (tl.cdiv(NT, 2) + 1)

    if TRANSPOSE_STATE:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    v += (bos * HV + i_hv).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    A += (bos * H + i_h).to(tl.int64) * BT
    beta += (bos * H + i_h).to(tl.int64)
    if FUSE_O:
        q += (bos * H + i_h).to(tl.int64) * K
        o += (bos * H + i_h).to(tl.int64) * V
    else:
        h += (boh * H + i_h).to(tl.int64) * K * V
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh.to(tl.int64) * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh.to(tl.int64) * K * V

    # load initial state
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            if TRANSPOSE_STATE:
                p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        if not FUSE_O:
            i_t_int64 = i_t.to(tl.int64)
            store_h = tl.constexpr(True)
            if not CKPT_PAIR:
                i_t_ckpt = i_t_int64
            elif not SKIP_STORE:
                if IS_VARLEN:
                    i_scratch = NT_CKPT - boh
                else:
                    i_scratch = NT_CKPT
                i_scratch_int64 = i_scratch.to(tl.int64)
                i_t_ckpt = tl.where(i_t_int64 % 2 == 0, i_t_int64 // 2, i_scratch_int64).to(tl.int64)
            else:
                i_t_ckpt = i_t_int64 // 2
                store_h = i_t_int64 % 2 == 0
            if store_h:
                if TRANSPOSE_STATE:
                    p_h1 = tl.make_block_ptr(h + i_t_ckpt * H * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
                else:
                    p_h1 = tl.make_block_ptr(h + i_t_ckpt * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
                if K > 64:
                    if TRANSPOSE_STATE:
                        p_h2 = tl.make_block_ptr(
                            h + i_t_ckpt * H * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                        )
                    else:
                        p_h2 = tl.make_block_ptr(
                            h + i_t_ckpt * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                        )
                    tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
                if K > 128:
                    if TRANSPOSE_STATE:
                        p_h3 = tl.make_block_ptr(
                            h + i_t_ckpt * H * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                        )
                    else:
                        p_h3 = tl.make_block_ptr(
                            h + i_t_ckpt * H * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                        )
                    tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
                if K > 192:
                    if TRANSPOSE_STATE:
                        p_h4 = tl.make_block_ptr(
                            h + i_t_ckpt * H * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                        )
                    else:
                        p_h4 = tl.make_block_ptr(
                            h + i_t_ckpt * H * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                        )
                    tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_u = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_u = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_u += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_u += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_u += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_u += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_u += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_u += tl.dot(b_w, b_h4.to(b_w.dtype))

        p_b = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_b = tl.load(p_b, boundary_check=(0,))

        p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        b_A = tl.load(p_A, boundary_check=(0, 1))

        p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A.to(b_vb.dtype), b_vb, allow_tf32=False) - b_u

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v_new, b_u.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        b_u = b_u.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        if FUSE_O:
            b_o = tl.zeros([BT, BV], dtype=tl.float32)
            b_P = tl.zeros([BT, BT], dtype=tl.float32)

            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))

            b_P += tl.dot(b_q, b_k)

        if TRANSPOSE_STATE:
            if FUSE_O:
                b_o += tl.dot(b_q, tl.trans(b_h1).to(b_q.dtype))
            b_h1 += tl.trans(tl.dot(b_k, b_u))
        else:
            if FUSE_O:
                b_o += tl.dot(b_q, b_h1.to(b_q.dtype))
            b_h1 += tl.dot(b_k, b_u)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if FUSE_O:
                p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_P += tl.dot(b_q, b_k)
            if TRANSPOSE_STATE:
                if FUSE_O:
                    b_o += tl.dot(b_q, tl.trans(b_h2).to(b_q.dtype))
                b_h2 += tl.trans(tl.dot(b_k, b_u))
            else:
                if FUSE_O:
                    b_o += tl.dot(b_q, b_h2.to(b_q.dtype))
                b_h2 += tl.dot(b_k, b_u)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if FUSE_O:
                p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_P += tl.dot(b_q, b_k)
            if TRANSPOSE_STATE:
                if FUSE_O:
                    b_o += tl.dot(b_q, tl.trans(b_h3).to(b_q.dtype))
                b_h3 += tl.trans(tl.dot(b_k, b_u))
            else:
                if FUSE_O:
                    b_o += tl.dot(b_q, b_h3.to(b_q.dtype))
                b_h3 += tl.dot(b_k, b_u)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if FUSE_O:
                p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_P += tl.dot(b_q, b_k)
            if TRANSPOSE_STATE:
                if FUSE_O:
                    b_o += tl.dot(b_q, tl.trans(b_h4).to(b_q.dtype))
                b_h4 += tl.trans(tl.dot(b_k, b_u))
            else:
                if FUSE_O:
                    b_o += tl.dot(b_q, b_h4.to(b_q.dtype))
                b_h4 += tl.dot(b_k, b_u)

        if FUSE_O:
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            m_P = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
            b_P = tl.where(m_P, b_P, 0)

            # to fix mma -> mma layout conversion
            # already solved by triton v3.2 or higher
            b_o = b_o * scale + tl.dot(b_P.to(b_u.dtype), b_u) * scale

            p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "FUSE_DV_LOCAL": lambda args: args["A_local"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV, "SKIP_STORE": skip_store}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([2, 3, 4] if check_shared_mem("ampere") else [1])
        for BV in ([32, 64] if check_shared_mem("ada") else [32])
        for skip_store in [True, False]
    ],
    key=[
        "H",
        "HV",
        "HO",
        "K",
        "V",
        "BT",
        "BV",
        "USE_G",
        "USE_GK",
        "USE_INITIAL_STATE",
        "USE_FINAL_STATE_GRADIENT",
        "USE_EXP2",
        "TRANSPOSE_STATE",
        "IS_VARLEN",
        "CKPT_PAIR",
        "FUSE_DV_LOCAL",
    ],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    A_local,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    NT_CKPT,
    H: tl.constexpr,
    HV: tl.constexpr,
    HO: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CKPT_PAIR: tl.constexpr,
    SKIP_STORE: tl.constexpr,
    FUSE_DV_LOCAL: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    i_ho = i_h // (HV // HO)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

        if not CKPT_PAIR:
            boh = i_n * NT
        else:
            # one dummy slot
            boh = i_n * (tl.cdiv(NT, 2) + 1)

    if TRANSPOSE_STATE:
        b_dh1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    do += (bos * HO + i_ho).to(tl.int64) * V
    if FUSE_DV_LOCAL:
        A_local += (bos * H + i_h // (HV // H)).to(tl.int64) * BT
    else:
        dv += (bos * HV + i_h).to(tl.int64) * V
    dv2 += (bos * HV + i_h).to(tl.int64) * V
    dh += (boh * HV + i_h).to(tl.int64) * K * V
    if USE_GK:
        gk += (bos * HV + i_h).to(tl.int64) * K

    if USE_INITIAL_STATE:
        dh0 += i_nh.to(tl.int64) * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh.to(tl.int64) * K * V

    if USE_FINAL_STATE_GRADIENT:
        if TRANSPOSE_STATE:
            p_dht1 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_dht2 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_dht3 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_dht4 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        store_dh = tl.constexpr(True)
        if not CKPT_PAIR:
            i_t_ckpt = i_t_int64
        else:
            is_ckpt = (i_t % 2 == 1) | (i_t == NT - 1)
            if not SKIP_STORE:
                if IS_VARLEN:
                    i_scratch = NT_CKPT - boh
                else:
                    i_scratch = NT_CKPT
                i_scratch_int64 = i_scratch.to(tl.int64)
                i_t_ckpt = tl.where(is_ckpt, i_t_int64 // 2, i_scratch_int64).to(tl.int64)
            else:
                i_t_ckpt = i_t_int64 // 2
                store_dh = is_ckpt
        if store_dh:
            if TRANSPOSE_STATE:
                p_dh1 = tl.make_block_ptr(dh + i_t_ckpt * HV * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
            else:
                p_dh1 = tl.make_block_ptr(dh + i_t_ckpt * HV * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                if TRANSPOSE_STATE:
                    p_dh2 = tl.make_block_ptr(
                        dh + i_t_ckpt * HV * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                    )
                else:
                    p_dh2 = tl.make_block_ptr(
                        dh + i_t_ckpt * HV * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                    )
                tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                if TRANSPOSE_STATE:
                    p_dh3 = tl.make_block_ptr(
                        dh + i_t_ckpt * HV * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                    )
                else:
                    p_dh3 = tl.make_block_ptr(
                        dh + i_t_ckpt * HV * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                    )
                tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                if TRANSPOSE_STATE:
                    p_dh4 = tl.make_block_ptr(
                        dh + i_t_ckpt * HV * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                    )
                else:
                    p_dh4 = tl.make_block_ptr(
                        dh + i_t_ckpt * HV * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                    )
                tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        last_idx = (min((i_t + 1) * BT, T) - 1).to(tl.int64)
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) * HV + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)
            else:
                bg_last_exp = exp(bg_last)
                b_g_exp = exp(b_g)

        if not FUSE_DV_LOCAL:
            p_dv = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (HO * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))

        # Update dv
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + last_idx * HV * K + o_k1, mask=(o_k1 < K), other=0.0).to(tl.float32)
        if TRANSPOSE_STATE:
            b_dv = tl.dot(b_k, tl.trans(b_dh1).to(b_k.dtype))
        else:
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + last_idx * HV * K + o_k2, mask=(o_k2 < K), other=0.0).to(tl.float32)
            if TRANSPOSE_STATE:
                b_dv += tl.dot(b_k, tl.trans(b_dh2).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + last_idx * HV * K + o_k3, mask=(o_k3 < K), other=0.0).to(tl.float32)
            if TRANSPOSE_STATE:
                b_dv += tl.dot(b_k, tl.trans(b_dh3).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + last_idx * HV * K + o_k4, mask=(o_k4 < K), other=0.0).to(tl.float32)
            if TRANSPOSE_STATE:
                b_dv += tl.dot(b_k, tl.trans(b_dh4).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            if USE_EXP2:
                b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
            else:
                b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
        if FUSE_DV_LOCAL:
            # dv_local is rebuilt in-kernel from A_local
            p_A_local = tl.make_block_ptr(A_local, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
            b_A_local = tl.load(p_A_local, boundary_check=(0, 1))
            b_dv += tl.dot(b_A_local, b_do)
        else:
            b_dv += tl.load(p_dv, boundary_check=(0, 1))

        tl.store(p_dv2, b_dv.to(p_dv2.dtype.element_ty), boundary_check=(0, 1))
        # Update dh
        p_w = tl.make_block_ptr(w, (K, T), (1, HV * K), (0, i_t * BT), (64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            if TRANSPOSE_STATE:
                if USE_EXP2:
                    b_dh1 *= exp2(b_gk_last1)[None, :]
                else:
                    b_dh1 *= exp(b_gk_last1)[None, :]
            else:
                if USE_EXP2:
                    b_dh1 *= exp2(b_gk_last1[:, None])
                else:
                    b_dh1 *= exp(b_gk_last1[:, None])
        if TRANSPOSE_STATE:
            b_dh1 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
        else:
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, HV * K), (64, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_dh2 *= exp2(b_gk_last2)[None, :]
                    else:
                        b_dh2 *= exp(b_gk_last2)[None, :]
                else:
                    if USE_EXP2:
                        b_dh2 *= exp2(b_gk_last2[:, None])
                    else:
                        b_dh2 *= exp(b_gk_last2[:, None])
            if TRANSPOSE_STATE:
                b_dh2 += tl.trans(
                    tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
                )
            else:
                b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, HV * K), (128, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_dh3 *= exp2(b_gk_last3)[None, :]
                    else:
                        b_dh3 *= exp(b_gk_last3)[None, :]
                else:
                    if USE_EXP2:
                        b_dh3 *= exp2(b_gk_last3[:, None])
                    else:
                        b_dh3 *= exp(b_gk_last3[:, None])
            if TRANSPOSE_STATE:
                b_dh3 += tl.trans(
                    tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
                )
            else:
                b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, HV * K), (192, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_dh4 *= exp2(b_gk_last4)[None, :]
                    else:
                        b_dh4 *= exp(b_gk_last4)[None, :]
                else:
                    if USE_EXP2:
                        b_dh4 *= exp2(b_gk_last4[:, None])
                    else:
                        b_dh4 *= exp(b_gk_last4[:, None])
            if TRANSPOSE_STATE:
                b_dh4 += tl.trans(
                    tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
                )
            else:
                b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_dh0 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_dh1 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_dh2 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_dh3 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


@dispatch("common")
def chunk_gated_delta_rule_fwd_h(
    q: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
    fuse_o: bool = False,
    ckpt_pair: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    assert H % HV == 0, f"H={H} must be a multiple of HV={HV}"
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
        NT_CKPT = triton.cdiv(NT, 2) if ckpt_pair else NT
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT * 2 if ckpt_pair else BT)
        NT_CKPT = len(prepare_chunk_indices(cu_seqlens, BT * 2)) if ckpt_pair else NT
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if ckpt_pair:
        NT_CKPT_alloc = NT_CKPT + 1
    else:
        NT_CKPT_alloc = NT_CKPT
    if transpose_state_layout:
        h = k.new_empty(B, NT_CKPT_alloc, H, V, K) if not fuse_o else None
        final_state = k.new_zeros(N, H, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT_CKPT_alloc, H, K, V) if not fuse_o else None
        final_state = k.new_zeros(N, H, K, V, dtype=torch.float32) if output_final_state else None

    o = v.new_empty(B, T, H, V) if fuse_o else None
    v_new = v.new_empty(B, T, H, V) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        q=q,
        k=k,
        v=v,
        beta=beta,
        A=A,
        w=w,
        v_new=v_new,
        o=o,
        h=h,
        h0=initial_state,
        ht=final_state,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        NT_CKPT=NT_CKPT,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        TRANSPOSE_STATE=transpose_state_layout,
        FUSE_O=fuse_o,
        CKPT_PAIR=ckpt_pair,
    )
    return o, h, v_new, final_state


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor | None,
    A_local: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
    ckpt_pair: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HO = *q.shape, do.shape[-1], do.shape[2]
    HV = H
    assert HV % HO == 0, f"HV={HV} must be a multiple of HO={HO}"
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
        NT_CKPT = triton.cdiv(NT, 2) if ckpt_pair else NT
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT * 2 if ckpt_pair else BT)
        NT_CKPT = len(prepare_chunk_indices(cu_seqlens, BT * 2)) if ckpt_pair else NT

    if ckpt_pair:
        NT_CKPT_alloc = NT_CKPT + 1
    else:
        NT_CKPT_alloc = NT_CKPT
    if transpose_state_layout:
        dh = q.new_empty(B, NT_CKPT_alloc, HV, V, K)
    else:
        dh = q.new_empty(B, NT_CKPT_alloc, HV, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    if A_local is None:
        assert dv is not None
        dv2 = torch.empty_like(dv)
    else:
        # dv_local is rebuilt in-kernel from A_local
        assert dv is None
        assert g is None
        assert gk is None
        assert A_local.shape == (B, T, H, BT)
        assert A_local.dtype == do.dtype == q.dtype
        dv2 = do.new_empty(B, T, HV, V, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * HV)

    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        A_local=A_local,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        NT_CKPT=NT_CKPT,
        H=H,
        HV=HV,
        HO=HO,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state_layout,
        CKPT_PAIR=ckpt_pair,
    )
    return dh, dh0, dv2


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [64, 128]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["B", "H", "HO", "K", "V", "BT", "TRANSPOSE_STATE", "IS_VARLEN"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dqw_pair(
    k,
    v_new,
    h,
    do,
    dq,
    dw,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    HO: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # reordered launch grid
    i_bh, i_k, i_t_ckpt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_ho = i_h // (H // HO)
    if IS_VARLEN:
        i_tg_ckpt = i_t_ckpt
        i_n, i_t_ckpt = tl.load(chunk_indices + i_t_ckpt * 2).to(tl.int32), tl.load(
            chunk_indices + i_t_ckpt * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        NT_CKPT = tl.cdiv(T, BT * 2)
        # +1 skips each batch row's trailing scratch slot in h/dh
        i_tg_ckpt = i_b * (NT_CKPT + 1) + i_t_ckpt
        bos, eos = i_b * T, i_b * T + T

    i_t0 = i_t_ckpt * 2
    i_t1 = i_t0 + 1

    # offset calculation
    v_new += (bos * H + i_h).to(tl.int64) * V
    do += (bos * HO + i_ho).to(tl.int64) * V
    h += (i_tg_ckpt * H + i_h).to(tl.int64) * K * V
    k += (bos * H + i_h).to(tl.int64) * K
    dq += (bos * H + i_h).to(tl.int64) * K
    dw += (bos * H + i_h).to(tl.int64) * K
    dv += (bos * H + i_h).to(tl.int64) * V

    b_dq0 = tl.zeros([BT, BK], dtype=tl.float32)
    b_dq1 = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds0 = tl.zeros([BT, BT], dtype=tl.float32)
    b_ds1 = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw0 = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw1 = tl.zeros([BT, BK], dtype=tl.float32)

    p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t0 * BT, i_k * BK), (BT, BK), (1, 0))
    p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t1 * BT, i_k * BK), (BT, BK), (1, 0))
    b_k0 = tl.load(p_k0, boundary_check=(0, 1))
    b_k1 = tl.load(p_k1, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_vn0 = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t0 * BT, i_v * BV), (BT, BV), (1, 0))
        p_vn1 = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t1 * BT, i_v * BV), (BT, BV), (1, 0))
        p_do0 = tl.make_block_ptr(do, (T, V), (HO * V, 1), (i_t0 * BT, i_v * BV), (BT, BV), (1, 0))
        p_do1 = tl.make_block_ptr(do, (T, V), (HO * V, 1), (i_t1 * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv0 = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t0 * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv1 = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t1 * BT, i_v * BV), (BT, BV), (1, 0))
        if TRANSPOSE_STATE:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_vn0 = tl.load(p_vn0, boundary_check=(0, 1))
        b_vn1 = tl.load(p_vn1, boundary_check=(0, 1))
        b_do0 = tl.load(p_do0, boundary_check=(0, 1))
        b_do1 = tl.load(p_do1, boundary_check=(0, 1))
        b_h0 = tl.load(p_h, boundary_check=(0, 1))
        # h[c1] = h[c0] + k[c0]^T v_new[c0], in the loaded [BV, BK] (h^T) orientation
        b_h1 = b_h0 + tl.dot(tl.trans(b_vn0), b_k0)
        b_ds0 += tl.dot(b_do0, tl.trans(b_vn0))
        b_ds1 += tl.dot(b_do1, tl.trans(b_vn1))
        b_dq0 += tl.dot(b_do0, b_h0.to(b_do0.dtype))
        b_dq1 += tl.dot(b_do1, b_h1.to(b_do1.dtype))
        b_dv0 = tl.load(p_dv0, boundary_check=(0, 1))
        b_dv1 = tl.load(p_dv1, boundary_check=(0, 1))
        b_dw0 += tl.dot(b_dv0, b_h0.to(b_dv0.dtype))
        b_dw1 += tl.dot(b_dv1, b_h1.to(b_dv1.dtype))

    p_dw0 = tl.make_block_ptr(dw, (T, K), (H * K, 1), (i_t0 * BT, i_k * BK), (BT, BK), (1, 0))
    p_dw1 = tl.make_block_ptr(dw, (T, K), (H * K, 1), (i_t1 * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dw0, -b_dw0.to(p_dw0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dw1, -b_dw1.to(p_dw1.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    o_t0 = i_t0 * BT + tl.arange(0, BT)
    o_t1 = i_t1 * BT + tl.arange(0, BT)
    m_t0 = o_t0 < T
    m_t1 = o_t1 < T
    m_A0 = (o_t0[:, None] >= o_t0[None, :]) & (m_t0[:, None] & m_t0)
    m_A1 = (o_t1[:, None] >= o_t1[None, :]) & (m_t1[:, None] & m_t1)
    b_ds0 = tl.where(m_A0, b_ds0, 0)
    b_ds1 = tl.where(m_A1, b_ds1, 0)
    b_ds0 = b_ds0.to(b_k0.dtype)
    b_ds1 = b_ds1.to(b_k1.dtype)
    b_dq0 += tl.dot(b_ds0, b_k0)
    b_dq1 += tl.dot(b_ds1, b_k1)
    b_dq0 *= scale
    b_dq1 *= scale
    p_dq0 = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t0 * BT, i_k * BK), (BT, BK), (1, 0))
    p_dq1 = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t1 * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq0, b_dq0.to(p_dq0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq1, b_dq1.to(p_dq1.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [64, 128]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["B", "H", "HO", "K", "V", "BT", "TRANSPOSE_STATE", "IS_VARLEN"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dk_pair(
    q,
    w,
    v_new,
    do,
    dh,
    dk,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    HO: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # reordered launch grid
    i_bh, i_k, i_t_ckpt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_ho = i_h // (H // HO)
    if IS_VARLEN:
        i_tg_ckpt = i_t_ckpt
        i_n, i_t_ckpt = tl.load(chunk_indices + i_t_ckpt * 2).to(tl.int32), tl.load(
            chunk_indices + i_t_ckpt * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        NT_CKPT = tl.cdiv(T, BT * 2)
        # +1 skips each batch row's trailing scratch slot in h/dh
        i_tg_ckpt = i_b * (NT_CKPT + 1) + i_t_ckpt
        bos, eos = i_b * T, i_b * T + T

    i_t0 = i_t_ckpt * 2
    i_t1 = i_t0 + 1

    # offset calculation
    v_new += (bos * H + i_h).to(tl.int64) * V
    do += (bos * HO + i_ho).to(tl.int64) * V
    dh += (i_tg_ckpt * H + i_h).to(tl.int64) * K * V
    q += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    dk += (bos * H + i_h).to(tl.int64) * K
    dv += (bos * H + i_h).to(tl.int64) * V

    b_dk0 = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk1 = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds0 = tl.zeros([BT, BT], dtype=tl.float32)
    b_ds1 = tl.zeros([BT, BT], dtype=tl.float32)

    p_q0 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t0 * BT, i_k * BK), (BT, BK), (1, 0))
    p_q1 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t1 * BT, i_k * BK), (BT, BK), (1, 0))
    p_w1 = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t1 * BT, i_k * BK), (BT, BK), (1, 0))
    b_q0 = tl.load(p_q0, boundary_check=(0, 1))
    b_q1 = tl.load(p_q1, boundary_check=(0, 1))
    b_w1 = tl.load(p_w1, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_vn0 = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t0 * BT, i_v * BV), (BT, BV), (1, 0))
        p_vn1 = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t1 * BT, i_v * BV), (BT, BV), (1, 0))
        p_do0 = tl.make_block_ptr(do, (T, V), (HO * V, 1), (i_t0 * BT, i_v * BV), (BT, BV), (1, 0))
        p_do1 = tl.make_block_ptr(do, (T, V), (HO * V, 1), (i_t1 * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv1 = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t1 * BT, i_v * BV), (BT, BV), (1, 0))
        if TRANSPOSE_STATE:
            p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_vn0 = tl.load(p_vn0, boundary_check=(0, 1))
        b_vn1 = tl.load(p_vn1, boundary_check=(0, 1))
        b_do0 = tl.load(p_do0, boundary_check=(0, 1))
        b_do1 = tl.load(p_do1, boundary_check=(0, 1))
        b_dv1 = tl.load(p_dv1, boundary_check=(0, 1))
        b_dh1 = tl.load(p_dh, boundary_check=(0, 1))
        # dh[c0] = dh[c1] + q[c1]^T do[c1] * scale - w[c1]^T dv2[c1], in [BV, BK] (dh^T) orientation
        b_dh0 = b_dh1 + tl.dot(tl.trans(b_do1), b_q1) * scale - tl.dot(tl.trans(b_dv1).to(b_w1.dtype), b_w1)
        b_ds0 += tl.dot(b_do0, tl.trans(b_vn0))
        b_ds1 += tl.dot(b_do1, tl.trans(b_vn1))
        b_dk0 += tl.dot(b_vn0, b_dh0.to(b_vn0.dtype))
        b_dk1 += tl.dot(b_vn1, b_dh1.to(b_vn1.dtype))

    tl.debug_barrier()
    o_t0 = i_t0 * BT + tl.arange(0, BT)
    o_t1 = i_t1 * BT + tl.arange(0, BT)
    m_t0 = o_t0 < T
    m_t1 = o_t1 < T
    m_A0 = (o_t0[:, None] >= o_t0[None, :]) & (m_t0[:, None] & m_t0)
    m_A1 = (o_t1[:, None] >= o_t1[None, :]) & (m_t1[:, None] & m_t1)
    b_ds0 = tl.where(m_A0, b_ds0, 0)
    b_ds1 = tl.where(m_A1, b_ds1, 0)
    b_ds0 = b_ds0.to(b_q0.dtype)
    b_ds1 = b_ds1.to(b_q1.dtype)
    b_dk0 += tl.dot(tl.trans(b_ds0), b_q0) * scale
    b_dk1 += tl.dot(tl.trans(b_ds1), b_q1) * scale
    p_dk0 = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t0 * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk1 = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t1 * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dk0, b_dk0.to(p_dk0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk1, b_dk1.to(p_dk1.dtype.element_ty), boundary_check=(0, 1))


def chunk_bwd_dqkw_pair(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    v_new: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HO = *k.shape, v_new.shape[-1], do.shape[2]
    assert H % HO == 0, f"H={H} must be a multiple of HO={HO}"
    BT = chunk_size
    if cu_seqlens is None:
        chunk_indices_pair = None
        NT_CKPT = triton.cdiv(T, BT * 2)
    else:
        chunk_indices_pair = prepare_chunk_indices(cu_seqlens, BT * 2)
        NT_CKPT = len(chunk_indices_pair)

    dq = q.new_empty(B, T, H, K)
    dk = k.new_empty(B, T, H, K)
    dw = torch.empty_like(w, dtype=torch.float32)

    # reorder the launch so `head` dimension moves fastest
    # to take advantages of L2 cache reuse when heads are tied
    def grid(meta):
        return (B * H, triton.cdiv(K, meta["BK"]), NT_CKPT)

    chunk_bwd_kernel_dk_pair[grid](
        q=q,
        w=w,
        v_new=v_new,
        do=do,
        dh=dh,
        dk=dk,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices_pair,
        scale=scale,
        B=B,
        T=T,
        H=H,
        HO=HO,
        K=K,
        V=V,
        BT=BT,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    chunk_bwd_kernel_dqw_pair[grid](
        k=k,
        v_new=v_new,
        h=h,
        do=do,
        dw=dw,
        dq=dq,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices_pair,
        scale=scale,
        B=B,
        T=T,
        H=H,
        HO=HO,
        K=K,
        V=V,
        BT=BT,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    return dq, dk, dw
