# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.numpy as jnp

from ....math import ceil_divide
from .backward import _linear_attention_backward_core
from .forward import _BATCHED_HEAD_GROUP, _linear_attention_forward_core
from .state_passing import _linear_attention_state_passing_core


_MAX_HEADS_PER_PALLAS_CELL = 16


def _cumulative_log_decay(log_f: jax.Array | None, BLOCK_SIZE_S: int) -> jax.Array | None:
    if log_f is None:
        return None

    B, S = log_f.shape[:2]
    assert S % BLOCK_SIZE_S == 0

    log_f = log_f.reshape(B, S // BLOCK_SIZE_S, BLOCK_SIZE_S, *log_f.shape[2:])
    log_f = log_f.astype(jnp.float32)
    log_f_cumsum = jnp.cumsum(log_f, axis=2)
    log_f_cumsum = log_f_cumsum.reshape(B, S, *log_f.shape[3:])

    return log_f_cumsum


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _linear_attention_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f: jax.Array | None,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    if log_f is not None and log_f.ndim == 4:
        Nq = q.shape[2]
        Nk = k.shape[2]
        Nv = v.shape[2]
        Nf = log_f.shape[2]
        N = max(Nq, Nk, Nv, log_f.shape[2])

        is_multi_head = all([i == N for i in (Nq, Nk, Nv, Nf)])
        fused_scan = is_multi_head and N % _BATCHED_HEAD_GROUP == 0
        log_f_cumsum = log_f
    else:
        fused_scan = False
        log_f_cumsum = _cumulative_log_decay(log_f, BLOCK_SIZE_S)

    y, ht = _linear_attention_forward_core(
        q=q,
        k=k,
        v=v,
        log_f_cumsum=log_f_cumsum,
        h0=h0,
        attention_multiplier=attention_multiplier,
        output_state=output_state,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        fused_scan=fused_scan,
    )

    return y, ht


def _linear_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f: jax.Array | None,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[tuple[jax.Array, jax.Array | None], tuple]:
    fused_scan = False
    if log_f is not None and log_f.ndim == 4:
        Nq_, Nk_, Nv_ = q.shape[2], k.shape[2], v.shape[2]
        N_ = max(Nq_, Nk_, Nv_, log_f.shape[2])
        fused_scan = Nq_ == N_ and Nk_ == N_ and Nv_ == N_ and log_f.shape[2] == N_ and N_ % _BATCHED_HEAD_GROUP == 0

    log_f_cumsum = log_f if fused_scan else _cumulative_log_decay(log_f, BLOCK_SIZE_S)

    y, h = _linear_attention_forward_core(
        q=q,
        k=k,
        v=v,
        log_f_cumsum=log_f_cumsum,
        h0=h0,
        attention_multiplier=attention_multiplier,
        output_state=output_state,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        fused_scan=fused_scan,
    )

    return (y, h), (q, k, v, log_f_cumsum, h0)


@partial(jax.jit, static_argnames=("attention_multiplier", "output_state", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward(
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    residuals: tuple,
    cotangents: tuple,
) -> tuple:
    q, k, v, log_f_cumsum, h0 = residuals
    dy, dht = cotangents
    dht = dht if output_state else None

    B, S, Nq, K = q.shape
    Nk = k.shape[2]
    Nv, V = v.shape[-2:]
    Nf = 0 if log_f_cumsum is None else log_f_cumsum.shape[2]

    N = max(Nq, Nk, Nv, Nf)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    # a 4-d log_f_cumsum residual on the same batched-layout hypothesis as the VJP
    # forward is RAW log_f on the fused path (see _linear_attention_forward).
    fused_scan = (
        log_f_cumsum is not None
        and log_f_cumsum.ndim == 4
        and Nq == N
        and Nk == Nv == N
        and log_f_cumsum.shape[2] == N
        and N % _BATCHED_HEAD_GROUP == 0
    )

    h = _linear_attention_state_passing_core(
        k=k,
        v=v,
        log_f_cumsum=log_f_cumsum,
        h0=h0,
        N=N,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        fused_scan=fused_scan,
    )

    # batched bwd kernel: same eligibility as the forward's batched path (un-gated,
    # scalar-shared-gate, or diagonal raw/cumsum log_f with full-head layout).
    batched_bwd = (
        Nq == N
        and Nk == Nv == N
        and N % _BATCHED_HEAD_GROUP == 0
        and (
            log_f_cumsum is None
            or (log_f_cumsum.ndim == 3 and log_f_cumsum.shape[2] == 1)
            # diagonal: only the fused full-head layout (raw log_f; a chunked cross-head-shared
            # diagonal gate arrives as ndim==4 with Nf == 1 and must keep the per-head kernel)
            or (log_f_cumsum.ndim == 4 and log_f_cumsum.shape[2] == N and fused_scan)
        )
    )

    dq, dk, dv, dlog_f, dh0 = _linear_attention_backward_core(
        q=q,
        k=k,
        v=v,
        log_f_cumsum=log_f_cumsum,
        h=h,
        dy=dy,
        dht=dht,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        fused_scan=fused_scan,
        batched=batched_bwd,
    )

    dq = dq.reshape(B, S, Nq, Gq, K).sum(axis=3)
    dk = dk.reshape(B, S, Nk, Gk, K).sum(axis=3)
    dv = dv.reshape(B, S, Nv, Gv, V).sum(axis=3)

    if log_f_cumsum is not None:
        Gf = N // Nf
        dlog_f = dlog_f.reshape(B, S, Nf, Gf, *dlog_f.shape[3:]).sum(axis=3)

    if h0 is None:
        dh0 = None

    return dq, dk, dv, dlog_f, dh0


_linear_attention_pallas.defvjp(_linear_attention_forward, _linear_attention_backward)


def _linear_attention_pallas_chunked(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f: jax.Array | None,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    Nq = q.shape[-2]
    Nk = k.shape[-2]
    Nv = v.shape[-2]
    Nf = 0 if log_f is None else log_f.shape[2]

    N = max(Nq, Nk, Nv, Nf)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv
    Gf = None if log_f is None else N // Nf

    groups = [(Gq, "query"), (Gk, "key"), (Gv, "value")]
    if log_f is not None:
        groups.append((Gf, "log_forget"))

    for G, name in groups:
        if _MAX_HEADS_PER_PALLAS_CELL % G != 0:
            raise ValueError(
                f"grouped head layout with a {name} group size of {G} cannot be split across "
                f"{_MAX_HEADS_PER_PALLAS_CELL}-head chunks (N={N}, Nq={Nq}, Nk={Nk}, Nv={Nv}, Nf={Nf}); "
                "choose q/k/v head counts whose group sizes all divide "
                f"{_MAX_HEADS_PER_PALLAS_CELL}, or use KernelBackend.jax"
            )

    NUM_CHUNKS = ceil_divide(N, _MAX_HEADS_PER_PALLAS_CELL)

    y = []
    ht = []

    for i in range(NUM_CHUNKS):
        start = i * _MAX_HEADS_PER_PALLAS_CELL
        end = min(N, start + _MAX_HEADS_PER_PALLAS_CELL)

        _log_f = None
        if log_f is not None:
            head_slice = slice(start // Gf, end // Gf)
            _log_f = log_f[..., head_slice] if log_f.ndim == 3 else log_f[..., head_slice, :]

        _y, _ht = _linear_attention_pallas(
            q=q[..., start // Gq : end // Gq, :],
            k=k[..., start // Gk : end // Gk, :],
            v=v[..., start // Gv : end // Gv, :],
            log_f=_log_f,
            h0=None if h0 is None else h0[:, start:end],
            attention_multiplier=attention_multiplier,
            output_state=output_state,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
        )

        y.append(_y)
        ht.append(_ht)

    y = jnp.concatenate(y, axis=2)
    ht = jnp.concatenate(ht, axis=1) if output_state else None

    return y, ht
