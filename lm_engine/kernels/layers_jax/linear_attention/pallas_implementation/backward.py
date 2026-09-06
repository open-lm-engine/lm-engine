# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide
from .forward import _BATCHED_HEAD_GROUP, _get_causal_mask


# Round-23 negative result (2026-08-31, tune_bwd_vscale.json / bench_bwd_vscale.py): a knob
# making the decay-to-operand scalings (qf/kf_inv/kf_last, MXU inputs) multiply in bf16 (f
# pre-cast instead of f32-mul-then-cast) produced BITWISE-IDENTICAL gradients on both gated
# lanes and identical wall/tune (1489.5 vs 1490.4 scalar, 1877.3 vs 1877.3 diag) — Mosaic
# already pattern-matches the f32-scaling->bf16-cast to the same narrow multiply. The VALU
# pressure (LLO: VALU 70%, VPOP 32-35%, VSTORE+spill 38-45%, MXU ~50%) is not in those
# multiplies. Knob removed; no shipped code changed.


# batched backward: same block layout as _linear_attention_backward_kernel, but heads are
# processed _BATCHED_HEAD_GROUP-at-a-time with a leading head batch dim on every dot
# (mirrors the forward's batched kernel): 4x fewer single-head matmul rounds per cell.
# Eligibility (host-side): Nq == Nk == Nv == N, Nf in {None, 1, N}, N % G == 0.
def _linear_attention_backward_batched_kernel(
    q_ref,
    k_ref,
    v_ref,
    log_f_cumsum_ref,
    h_ref,
    dy_ref,
    dht_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    df_ref,
    dh0_ref,
    dh_scratch,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    N: int,
    f_diagonal: bool,
    NUM_BLOCKS_V: int,
    NUM_BLOCKS_S: int,
    fused_scan: bool,
) -> None:
    S_CELLS_VISITED = pl.program_id(1)

    @pl.when(S_CELLS_VISITED == 0)
    def _():
        if dht_ref is None:
            dh_scratch[...] = jnp.zeros_like(dh_scratch)
        else:
            dh_scratch[...] = dht_ref[...].astype(jnp.float32)

    dtype = q_ref.dtype
    G = _BATCHED_HEAD_GROUP

    q_ = q_ref[...].transpose(1, 0, 2)
    k_ = k_ref[...].transpose(1, 0, 2)
    v_ = v_ref[...].transpose(1, 0, 2)
    dy_ = dy_ref[...].transpose(1, 0, 2)
    h_ = h_ref[...].astype(dtype)

    log_f_cumsum_ = None
    if log_f_cumsum_ref is not None:
        log_f_cumsum_ = log_f_cumsum_ref[...]

    causal_mask = _get_causal_mask(BLOCK_SIZE_S)
    K = q_.shape[-1]

    TriL_batched = None
    if log_f_cumsum_ref is not None and f_diagonal and fused_scan:
        TriL_batched = jnp.tril(jnp.ones((G, BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32))

    TriU_batched = None
    if log_f_cumsum_ref is not None:
        TriU_batched = jnp.triu(jnp.ones((G, BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32))

    if log_f_cumsum_ref is not None and not f_diagonal:
        log_f = log_f_cumsum_[:, 0].astype(jnp.float32)
        # Round 24a negative result: factoring f_ij_scalar = exp(a-b) as exp(a)*exp(-b)
        # (outer product of two (BS,) vectors; the scalar residual is chunk-local so the
        # bound class is unchanged) measured flat-equal tune (1492.1 vs 1490.1) — XLA/Mosaic
        # already handles the (BS,BS) exp cheaply. Round 24b negative result: folding the
        # decay into DOT OPERANDS (kf_inv=k*exp(-c) for qk; decay-scaled dy/v copies for
        # dyv; raw k/q intra dots) REGRESSED (+1.5%: 1513.0) — extra bf16 operand copies
        # feed the spill budget (LLO spill fractions 38-45%) — and DOUBLED dq rel error
        # (0.0089 vs 0.0044 vs gold) from pre-MXU bf16 rounding of the scaled operands.
        # Reverted to the shipped f32 matrix form below.
        f_ij_scalar = jnp.exp(log_f[:, None] - log_f[None, :])
        f_scalar = jnp.exp(log_f)
        f_last_scalar = jnp.exp(log_f[-1])
        f_to_last_scalar = jnp.exp(log_f[-1] - log_f)

    for n in range(0, N, G):
        g = slice(n, n + G)
        q = q_[g].astype(dtype)
        k = k_[g].astype(dtype)
        dy_full = (dy_[g].astype(jnp.float32) * attention_multiplier).astype(dtype)

        if log_f_cumsum_ is None:
            qk = jax.lax.dot_general(q, k, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            qk = jnp.where(causal_mask[None], qk, 0).astype(dtype)

            qf = q
            kf_last = k
            kf_inv = None
        else:
            if f_diagonal:
                log_f_g = log_f_cumsum_[:, g, :].transpose(1, 0, 2).astype(jnp.float32)
                if fused_scan:
                    log_f_g = jax.lax.dot_general(
                        TriL_batched, log_f_g, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
                    )
                log_f_last = log_f_g[:, -1]  # (G, K)

                f = jnp.exp(log_f_g)
                f_last = jnp.exp(log_f_last)  # (G, K)
                f_inv = jnp.exp(-log_f_g)
                # Round 23c: on the fused diagonal lane all window cumsums are chunk-local
                # (TriL block-scanned residual), so exp(c_last - c) == f_last * f_inv exactly;
                # trading a (G, BS, K) exp for a broadcast multiply relieves the VALU (LLO:
                # VALU-bound at ~70%). The non-fused diagonal lane consumes GLOBAL cumsums,
                # where factored exp must not overflow/underflow the same range, so it keeps
                # the direct exp.
                f_to_last = f_last[:, None, :] * f_inv if fused_scan else jnp.exp(log_f_last[:, None, :] - log_f_g)

                qf = (q * f).astype(dtype)
                kf_inv = (k * f_inv).astype(dtype)
                kf_last = (k * f_to_last).astype(dtype)

                qk = jax.lax.dot_general(qf, kf_inv, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            else:
                f_to_last_b = f_to_last_scalar
                f_b = f_scalar

                qf = (q * f_b[None, :, None]).astype(dtype)
                kf_last = (k * f_to_last_b[None, :, None]).astype(dtype)
                kf_inv = None

                qk = jax.lax.dot_general(q, k, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
                qk = qk * f_ij_scalar[None]

            qk = jnp.where(causal_mask[None], qk, 0).astype(dtype)

        # Round-24c negative result: num_blocks_v==1 bf16 dyv narrowing (halving its live
        # VMEM) regressed scalar +2.0% (bf16 dyv makes the f_ij post-multiply promote to
        # f32 and cast back) and was flat on diag/none (1155.4/1835.1). f32 dyv restored.
        dyv = jnp.zeros((G, BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32)
        dq = jnp.zeros((G, BLOCK_SIZE_S, K), jnp.float32)
        dk = jnp.zeros((G, BLOCK_SIZE_S, K), jnp.float32)

        if log_f_cumsum_ is not None:
            df = jnp.zeros((G, BLOCK_SIZE_S, K), jnp.float32)
            df_last = jnp.zeros((G, K), jnp.float32)

        # pallas refs support one python-slice index level (dh_scratch[g]) but not nested
        # [g][:, :, s:e]; materialize the group window, then write the full window back once.
        dh_group = dh_scratch[g][...]
        dh_out = None

        for BLOCK_ID_V in range(NUM_BLOCKS_V):
            start = BLOCK_ID_V * BLOCK_SIZE_V
            end = start + BLOCK_SIZE_V

            v = v_[g][:, :, start:end].astype(dtype)
            dy = dy_full[:, :, start:end]
            h = h_[g][:, :, start:end]
            dh = dh_group[:, :, start:end]

            # Round-26 negative (none-lane dv pair): fusing the two dv dots into ONE by
            # concatenating along the contracted axis (cat([qk,kf_last],2) @ cat([dy;dh],1))
            # measured 2015.8 tuning-us standalone none core S=4096 vs 1004.2 (2x SLOWER —
            # the per-window (G,256,384)+(G,384,128) bf16 concats feed the copy/spill budget)
            # and broke the S=512 bitwise standalone-vs-VJP dv contract. Kept as two dots.
            dv = jax.lax.dot_general(qk, dy, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            dv += jax.lax.dot_general(
                kf_last, dh.astype(dtype), (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
            )
            dv_ref[:, g, start:end] = dv.transpose(1, 0, 2).astype(dtype)

            _dq = jax.lax.dot_general(dy, h, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            if log_f_cumsum_ is not None:
                _dq *= f if f_diagonal else f_b[None, :, None]
            dq += _dq

            _dk = jax.lax.dot_general(v, dh, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            if log_f_cumsum_ is not None:
                _dk *= f_to_last if f_diagonal else f_to_last_b[None, :, None]
            dk += _dk

            if log_f_cumsum_ is not None:
                df_q = q.astype(jnp.float32) * _dq
                df_k = k.astype(jnp.float32) * _dk
                df_decay = h.astype(jnp.float32) * dh

                df += df_q - df_k
                df_last += df_k.sum(axis=1) + (f_last if f_diagonal else f_last_scalar) * df_decay.sum(axis=2)

            if log_f_cumsum_ is not None:
                dh *= f_last[:, :, None] if f_diagonal else f_last_scalar
            dh += jax.lax.dot_general(qf, dy, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            dh_out = dh if dh_out is None else jnp.concatenate((dh_out, dh), axis=2)

            dyv += jax.lax.dot_general(dy, v, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)

        dh_scratch[g] = dh_out

        dyv = jnp.where(causal_mask[None], dyv, 0)
        if log_f_cumsum_ is not None and not f_diagonal:
            dyv *= f_ij_scalar[None]
        dyv = dyv.astype(dtype)

        dq_intra = jax.lax.dot_general(
            dyv, k if kf_inv is None else kf_inv, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
        )

        dk_intra = jax.lax.dot_general(
            dyv, q if kf_inv is None else qf, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
        )

        if f_diagonal:
            dq_intra *= f
            dk_intra *= f_inv

        dq += dq_intra
        dk += dk_intra

        dq_ref[:, g, :] = dq.transpose(1, 0, 2).astype(dtype)
        dk_ref[:, g, :] = dk.transpose(1, 0, 2).astype(dtype)

        if log_f_cumsum_ is not None:
            df += q.astype(jnp.float32) * dq_intra - k.astype(jnp.float32) * dk_intra
            dlog_f = (
                jax.lax.dot_general(TriU_batched, df, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
                + df_last[:, None, :]
            )

            if f_diagonal:
                df_ref[:, g, :] = dlog_f.transpose(1, 0, 2).astype(df_ref.dtype)
            else:
                df_ref[:, g] = dlog_f.sum(axis=-1).transpose(1, 0).astype(df_ref.dtype)

    @pl.when(S_CELLS_VISITED == NUM_BLOCKS_S - 1)
    def _():
        dh0_ref[...] = dh_scratch[...]


def _linear_attention_backward_kernel(
    q_ref,
    k_ref,
    v_ref,
    log_f_cumsum_ref,
    h_ref,
    dy_ref,
    dht_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    df_ref,
    dh0_ref,
    dh_scratch,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    N: int,
    Gq: int,
    Gk: int,
    Gv: int,
    Gf: int | None,
    f_diagonal: bool,
    NUM_BLOCKS_V: int,
    NUM_BLOCKS_S: int,
    fused_scan: bool,
) -> None:
    S_CELLS_VISITED = pl.program_id(1)

    @pl.when(S_CELLS_VISITED == 0)
    def _():
        if dht_ref is None:
            dh_scratch[...] = jnp.zeros_like(dh_scratch)
        else:
            dh_scratch[...] = dht_ref[...].astype(jnp.float32)

    dtype = q_ref.dtype

    q_ = q_ref[...].transpose(1, 0, 2)
    k_ = k_ref[...].transpose(1, 0, 2)
    v_ = v_ref[...].transpose(1, 0, 2)
    dy_ = dy_ref[...].transpose(1, 0, 2)
    h_ = h_ref[...].astype(dtype)

    log_f_cumsum_ = None
    if log_f_cumsum_ref is not None:
        log_f_cumsum_ = log_f_cumsum_ref[...]

        if f_diagonal:
            log_f_cumsum_ = log_f_cumsum_.transpose(1, 0, 2)
        else:
            log_f_cumsum_ = log_f_cumsum_.transpose(1, 0)

    causal_mask = _get_causal_mask(BLOCK_SIZE_S)
    K = q_.shape[-1]

    if fused_scan:
        # raw log_f in, chunk-local cumsum via triangular systolic matmul (see forward.py);
        # all chunk-products and df chains below are differences or chunk totals of the
        # cumsum, so the local scan is mathematically identical to the absolute one.
        TriL_scan = jnp.tril(jnp.ones((BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32))

    for n in range(N):
        q = q_[n // Gq].astype(dtype)
        k = k_[n // Gk].astype(dtype)
        dy_full = (dy_[n].astype(jnp.float32) * attention_multiplier).astype(dtype)

        if log_f_cumsum_ref is None:
            qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            qk = jnp.where(causal_mask, qk, 0).astype(dtype)

            qf = q
            kf_last = k
            kf_inv = None
        else:
            log_f = log_f_cumsum_[n // Gf].astype(jnp.float32)
            if fused_scan:
                log_f = jax.lax.dot_general(TriL_scan, log_f, (((1,), (0,)), ((), ())))
            log_f_last = log_f[-1]

            f = jnp.exp(log_f)
            f_last = jnp.exp(log_f_last)

            if f_diagonal:
                f_to_last = jnp.exp(log_f_last[None, :] - log_f)
                f_inv = jnp.exp(-log_f)

                qf = (q * f).astype(dtype)
                kf_inv = (k * f_inv).astype(dtype)
                kf_last = (k * f_to_last).astype(dtype)

                qk = jax.lax.dot_general(qf, kf_inv, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            else:
                f_to_last = jnp.exp(log_f_last - log_f)
                f_ij = jnp.exp(log_f[:, None] - log_f[None, :])

                qf = (q * f[:, None]).astype(dtype)
                kf_last = (k * f_to_last[:, None]).astype(dtype)
                kf_inv = None

                qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
                qk *= f_ij

            qk = jnp.where(causal_mask, qk, 0).astype(dtype)

        dyv = jnp.zeros((BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32)
        dq = jnp.zeros((BLOCK_SIZE_S, K), jnp.float32)
        dk = jnp.zeros((BLOCK_SIZE_S, K), jnp.float32)

        if log_f_cumsum_ is not None:
            df = jnp.zeros((BLOCK_SIZE_S, K), jnp.float32)
            df_last = jnp.zeros((K,), jnp.float32)

        for BLOCK_ID_V in range(NUM_BLOCKS_V):
            start = BLOCK_ID_V * BLOCK_SIZE_V
            end = start + BLOCK_SIZE_V

            v = v_[n // Gv][:, start:end].astype(dtype)
            dy = dy_full[:, start:end]
            h = h_[n][:, start:end]
            dh = dh_scratch[n][:, start:end]

            dv = jax.lax.dot_general(qk, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            dv += jax.lax.dot_general(
                kf_last, dh.astype(dtype), (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )
            dv_ref[:, n, start:end] = dv.astype(dtype)

            _dq = jax.lax.dot_general(dy, h, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            if log_f_cumsum_ is not None:
                _dq *= f if f_diagonal else f[:, None]
            dq += _dq

            _dk = jax.lax.dot_general(v, dh, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            if log_f_cumsum_ is not None:
                _dk *= f_to_last if f_diagonal else f_to_last[:, None]
            dk += _dk

            if log_f_cumsum_ is not None:
                df_q = q.astype(jnp.float32) * _dq
                df_k = k.astype(jnp.float32) * _dk
                df_decay = h.astype(jnp.float32) * dh

                df += df_q - df_k
                df_last += df_k.sum(axis=0) + f_last * df_decay.sum(axis=1)

            if log_f_cumsum_ is not None:
                dh *= f_last[:, None] if f_diagonal else f_last
            dh += jax.lax.dot_general(qf, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            dh_scratch[n, :, start:end] = dh

            dyv += jax.lax.dot_general(dy, v, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

        dyv = jnp.where(causal_mask, dyv, 0)
        if log_f_cumsum_ is not None and not f_diagonal:
            dyv *= f_ij
        dyv = dyv.astype(dtype)

        dq_intra = jax.lax.dot_general(
            dyv, k if kf_inv is None else kf_inv, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

        dk_intra = jax.lax.dot_general(
            dyv, q if kf_inv is None else qf, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

        if f_diagonal:
            dq_intra *= f
            dk_intra *= f_inv

        dq += dq_intra
        dk += dk_intra

        dq_ref[:, n, :] = dq.astype(dtype)
        dk_ref[:, n, :] = dk.astype(dtype)

        if log_f_cumsum_ is not None:
            suffix_sum = jnp.where(_get_causal_mask(BLOCK_SIZE_S, transpose=True), 1.0, 0.0)

            df += q.astype(jnp.float32) * dq_intra - k.astype(jnp.float32) * dk_intra
            dlog_f = jax.lax.dot_general(suffix_sum, df, (((1,), (0,)), ((), ()))) + df_last

            if f_diagonal:
                df_ref[:, n, :] = dlog_f.astype(df_ref.dtype)
            else:
                df_ref[:, n] = dlog_f.sum(axis=-1).astype(df_ref.dtype)

    @pl.when(S_CELLS_VISITED == NUM_BLOCKS_S - 1)
    def _():
        dh0_ref[...] = dh_scratch[...]


@partial(
    jax.jit,
    static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V", "fused_scan", "batched"),
)
def _linear_attention_backward_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f_cumsum: jax.Array | None,
    h: jax.Array,
    dy: jax.Array,
    dht: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    fused_scan: bool = False,
    batched: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    B, S, Nq, K = q.shape
    Nk = k.shape[2]
    Nv, V = v.shape[-2:]
    Nf = 0 if log_f_cumsum is None else log_f_cumsum.shape[2]
    N = dy.shape[2]

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv
    Gf = None if log_f_cumsum is None else N // Nf

    f_diagonal = log_f_cumsum is not None and log_f_cumsum.ndim == 4
    if log_f_cumsum is not None:
        assert log_f_cumsum.shape == (B, S, Nf, K) if f_diagonal else (B, S, Nf)

    assert S % BLOCK_SIZE_S == 0

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    NUM_BLOCKS_V = ceil_divide(V, BLOCK_SIZE_V)
    assert (
        V == NUM_BLOCKS_V * BLOCK_SIZE_V
    ), "V must be an integer multiple of BLOCK_SIZE_V (host padding guarantees this)"

    f_spec = None
    df_spec = None
    if log_f_cumsum is not None:
        if f_diagonal:
            f_spec = pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nf, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            )

            df_spec = pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            )
        else:
            f_spec = pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nf),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0),
            )

            df_spec = pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0),
            )

    target = (
        partial(
            _linear_attention_backward_batched_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            N=N,
            f_diagonal=f_diagonal,
            NUM_BLOCKS_V=NUM_BLOCKS_V,
            NUM_BLOCKS_S=NUM_BLOCKS_S,
            fused_scan=fused_scan,
        )
        if batched
        else partial(
            _linear_attention_backward_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            N=N,
            Gq=Gq,
            Gk=Gk,
            Gv=Gv,
            Gf=Gf,
            f_diagonal=f_diagonal,
            NUM_BLOCKS_V=NUM_BLOCKS_V,
            NUM_BLOCKS_S=NUM_BLOCKS_S,
            fused_scan=fused_scan,
        )
    )

    # Round-21 note: explicit input_output_aliases (dh0 over dht; df over dy) were tried
    # to let the diagonal core compile STANDALONE at S>=1024 and reverted — any explicit
    # alias overrides XLA's implicit composition-time choices (e.g. recycling the dead f32
    # raw-log_f buffer, which pallas cannot express cross-dtype) and makes the COMPOSED
    # pipeline E1001 instead. The shipped VJP relies on the implicit aliasing. For benchmark
    # isolation the standalone diagonal call instead compiles under the benchmark-only env
    # LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=40960" — verified shipped-path-
    # invariant (sweep deltas <=0.6%); see tune_bwd_core.py / OPTIMIZATION_LOG_R2 Round 21.
    kernel = pl.pallas_call(
        target,
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, S, N, K), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, S, N, K), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, S, N, V), dtype=q.dtype),
            (
                None
                if log_f_cumsum is None
                else jax.ShapeDtypeStruct(
                    shape=(B, S, N, K) if f_diagonal else (B, S, N),
                    # diagonal df in the operand dtype: the f32 variant is 537MB at S=4096 and
                    # single-handedly triggers E1001 scoped-vmem in the custom-call stack allocator;
                    # accumulation stays fp32 in-VMEM, only the HBM output narrows.
                    dtype=q.dtype if f_diagonal else jnp.float32,
                )
            ),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nq, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nk, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nv, V),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            f_spec,
            pl.BlockSpec(
                block_shape=(None, N, K, V),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, V),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            (
                None
                if dht is None
                else pl.BlockSpec(
                    block_shape=(None, N, K, V),
                    index_map=lambda B, _: (B, 0, 0, 0),
                )
            ),
        ),
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, V),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            df_spec,
            pl.BlockSpec(
                block_shape=(None, N, K, V),
                index_map=lambda B, S: (B, 0, 0, 0),
            ),
        ),
        scratch_shapes=[pltpu.VMEM((N, K, V), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, dimension_semantics=("parallel", "arbitrary")
        ),
    )

    return kernel(q, k, v, log_f_cumsum, h, dy, dht)
