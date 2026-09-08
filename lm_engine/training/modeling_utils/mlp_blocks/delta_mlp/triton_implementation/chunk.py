# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Literal

import torch
from fla.ops.cp import FLACPContext
from fla.ops.cp.chunk_delta_h import compress_h0, expand_h0

from .chunk_delta_h import chunk_bwd_dqkw_pair, chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from .chunk_delta_h_cp import chunk_gated_delta_rule_bwd_dhu_pre_process, chunk_gated_delta_rule_fwd_h_pre_process
from .chunk_fwd import chunk_gated_delta_rule_fwd_intra
from .chunk_o import chunk_bwd_A_local, chunk_bwd_dqkwg
from .wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd


def _maybe_pipeline_receive_initial_state(
    k: torch.Tensor,
    v_or_do: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    cp_context: FLACPContext,
    transpose_state_layout: bool,
    bwd: bool,
) -> torch.Tensor | None:
    # fwd receives state from the previous rank; bwd receives state from the next rank
    if bwd:
        is_pipeline_head = cp_context.is_last_rank
        rank_offset = 1
    else:
        is_pipeline_head = cp_context.is_first_rank
        rank_offset = -1

    if is_pipeline_head:
        initial_state_cp = initial_state
    else:
        B, _, H, K = k.shape
        _, _, _, V = v_or_do.shape
        if cu_seqlens is None:
            N = B
        else:
            N = len(cu_seqlens) - 1

        if transpose_state_layout:
            initial_state_cp = k.new_empty(N, H, V, K, dtype=torch.float32)
        else:
            initial_state_cp = k.new_empty(N, H, K, V, dtype=torch.float32)

        torch.distributed.recv(
            tensor=initial_state_cp,
            group=cp_context.group,
            group_src=torch.distributed.get_rank(group=cp_context.group) + rank_offset,
        )

    return initial_state_cp


def chunk_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
    cp_pipeline: bool = False,
    compute_in_fp32: bool | Literal["cp"] = False,
):
    assert beta.dtype == torch.float32
    assert q.shape[2] == k.shape[2] == beta.shape[2]
    assert q.shape[2] % v.shape[2] == 0
    if cp_context is not None:
        # i.e., batch size = 1
        assert cu_seqlens is not None
        assert cu_seqlens.ndim == 1
        assert cu_seqlens.shape[0] == 2

    # upstream FLA's `chunk_gated_delta_rule_fwd_kkt_solve_kernel` computes offset with int32
    assert k.numel() <= torch.iinfo(torch.int32).max

    if compute_in_fp32 == "cp":
        assert not cp_pipeline

    output_dtype = q.dtype
    if compute_in_fp32 is True:
        q = q.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)
        v = v.to(dtype=torch.float32)

    # obtain WY representation.
    # fused kkt + solve_tril + recompute_w_u
    w, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=None,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    assert A.dtype == torch.float32

    if cp_context is not None:
        if cp_pipeline:
            initial_state_cp = _maybe_pipeline_receive_initial_state(
                k=k,
                v_or_do=v,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
                transpose_state_layout=transpose_state_layout,
                bwd=False,
            )
        else:
            if initial_state is not None:
                assert initial_state.ndim == 4
                assert initial_state.shape[0] == 1
                initial_state = initial_state.squeeze(dim=0)

            dtype_cp_k = torch.float32 if compute_in_fp32 == "cp" else k.dtype
            dtype_cp_v = torch.float32 if compute_in_fp32 == "cp" else v.dtype
            dtype_cp_w = torch.float32 if compute_in_fp32 == "cp" else w.dtype
            initial_state_cp = chunk_gated_delta_rule_fwd_h_pre_process(
                k=k.to(dtype=dtype_cp_k),
                v=v.to(dtype=dtype_cp_v),
                w=w.to(dtype=dtype_cp_w),
                beta=beta,
                A=A,
                cu_seqlens=cu_seqlens,
                initial_state=initial_state,
                context=cp_context,
                transpose_state_layout=transpose_state_layout,
                fuse_u=True,
            )
    else:
        initial_state_cp = initial_state

    pipeline_send_final_state = cp_pipeline and cp_context is not None and not cp_context.is_last_rank
    o, _, _, final_state = chunk_gated_delta_rule_fwd_h(
        q=q,
        k=k,
        v=v,
        beta=beta,
        A=A,
        w=w,
        scale=scale,
        initial_state=initial_state_cp,
        output_final_state=output_final_state or pipeline_send_final_state,
        save_new_value=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
        fuse_o=True,
    )

    if pipeline_send_final_state:
        torch.distributed.send(
            tensor=final_state,
            group=cp_context.group,
            group_dst=torch.distributed.get_rank(group=cp_context.group) + 1,
        )
        final_state = final_state if output_final_state else None

    if cp_context is not None and not cp_pipeline:
        initial_state_cp = compress_h0(initial_state_cp, context=cp_context)

    o = o.to(dtype=output_dtype)
    return o, A, final_state, initial_state_cp


def _backward_dh_and_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    cp_context: FLACPContext | None,
    chunk_indices: torch.LongTensor | None,
    transpose_state_layout: bool,
    ckpt_pair: bool,
    cp_pipeline: bool,
    compute_in_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    A_local = chunk_bwd_A_local(
        q=q,
        k=k,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    if cp_context is not None:
        if cp_pipeline:
            dht = _maybe_pipeline_receive_initial_state(
                k=k,
                v_or_do=do,
                initial_state=dht,
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
                transpose_state_layout=transpose_state_layout,
                bwd=True,
            )
        else:
            if dht is not None:
                assert dht.ndim == 4
                assert dht.shape[0] == 1
                dht = dht.squeeze(dim=0)

            dtype_cp_q = torch.float32 if compute_in_fp32 else q.dtype
            dtype_cp_k = torch.float32 if compute_in_fp32 else k.dtype
            dtype_cp_w = torch.float32 if compute_in_fp32 else w.dtype
            dtype_cp_do = torch.float32 if compute_in_fp32 else do.dtype
            dtype_cp_A_local = torch.float32 if compute_in_fp32 else A_local.dtype
            dht = chunk_gated_delta_rule_bwd_dhu_pre_process(
                q=q.to(dtype=dtype_cp_q),
                k=k.to(dtype=dtype_cp_k),
                w=w.to(dtype=dtype_cp_w),
                do=do.to(dtype=dtype_cp_do),
                dv=None,
                A_local=A_local.to(dtype=dtype_cp_A_local),
                scale=scale,
                cu_seqlens=cu_seqlens,
                dht=dht,
                context=cp_context,
                transpose_state_layout=transpose_state_layout,
            )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=None,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=None,
        A_local=A_local,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
        ckpt_pair=ckpt_pair,
    )

    pipeline_send_dh0 = cp_pipeline and cp_context is not None and not cp_context.is_first_rank
    if pipeline_send_dh0:
        torch.distributed.send(
            tensor=dh0,
            group=cp_context.group,
            group_dst=torch.distributed.get_rank(group=cp_context.group) - 1,
        )

    return dh, dh0, dv, dht


def chunk_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
    cp_pipeline: bool = False,
    compute_in_fp32: bool | Literal["cp"] = False,
):
    assert beta.dtype == A.dtype == torch.float32
    assert q.shape[2] == k.shape[2] == beta.shape[2]
    assert q.shape[2] % v.shape[2] == 0
    if cp_context is not None:
        # i.e., batch size = 1
        assert cu_seqlens is not None
        assert cu_seqlens.ndim == 1
        assert cu_seqlens.shape[0] == 2

    if compute_in_fp32 == "cp":
        assert not cp_pipeline

    output_dtype = q.dtype
    if compute_in_fp32 is True:
        q = q.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)
        v = v.to(dtype=torch.float32)
        do = do.to(dtype=torch.float32)

    w, _ = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        compute_u=False,
    )

    if cp_context is not None and not cp_pipeline:
        initial_state = expand_h0(initial_state, context=cp_context)

    CKPT_PAIR = True
    dh, dh0, dv, dht = _backward_dh_and_dv(
        q=q,
        k=k,
        w=w,
        scale=scale,
        initial_state=initial_state,
        do=do,
        dht=dht,
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
        ckpt_pair=CKPT_PAIR,
        cp_pipeline=cp_pipeline,
        compute_in_fp32=(compute_in_fp32 == "cp"),
    )
    _, h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        q=None,
        k=k,
        v=v,
        beta=beta,
        A=A,
        w=w,
        scale=None,
        initial_state=initial_state,
        output_final_state=False,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
        fuse_o=False,
        ckpt_pair=CKPT_PAIR,
    )
    if CKPT_PAIR:
        # FIXME Han: commenting this out should be correct
        # assert chunk_indices is None
        dq, dk, dw = chunk_bwd_dqkw_pair(
            q=q,
            k=k,
            w=w,
            v_new=v_new,
            do=do,
            dv=dv,
            h=h,
            dh=dh,
            scale=scale,
            cu_seqlens=cu_seqlens,
            transpose_state_layout=transpose_state_layout,
        )
    else:
        dq, dk, dw, _ = chunk_bwd_dqkwg(
            q=q,
            k=k,
            v=v_new,
            w=w,
            g=None,
            h=h,
            dv=dv,
            do=do,
            dh=dh,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            transpose_state_layout=transpose_state_layout,
        )
    del h, dh
    dk2, dv, db, _ = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=None,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dk.add_(dk2)

    # Under CP, only rank 0's dh0_local is the true ∂L/∂h0_user (the bwd pre-process
    # merge already propagated later ranks' contributions back to rank 0 via dht
    # chaining). Non-first ranks' dh0_local is ∂L/∂(intermediate state), not w.r.t.
    # h0 — zero it out so DDP all-reduce-sum produces the correct gradient.
    if cp_context is not None:
        if (dh0 is not None) and (not cp_context.is_first_rank):
            if cp_pipeline:
                assert dht is not None or cp_context.is_last_rank
            else:
                assert dht is not None
            assert initial_state is not None
            dh0 = torch.zeros_like(dh0)

    dq = dq.to(dtype=output_dtype)
    dk = dk.to(dtype=output_dtype)
    dv = dv.to(dtype=output_dtype)
    return dq, dk, dv, db, dh0
