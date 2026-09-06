# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _linear_attention_state_passing_kernel(
    k_ref,
    v_ref,
    log_f_cumsum_ref,
    h0_ref,
    h_ref,
    h_scratch,
    N: int,
    Gk: int,
    Gv: int,
    Gf: int | None,
    f_diagonal: bool,
    NUM_BLOCKS_V: int,
    BLOCK_SIZE_V: int,
    fused_scan: bool,
    HEAD_GROUP: int,
) -> None:
    BLOCK_ID_S = pl.program_id(1)

    @pl.when(BLOCK_ID_S == 0)
    def _():
        if h0_ref is None:
            h_scratch[...] = jnp.zeros_like(h_scratch)
        else:
            h_scratch[...] = h0_ref[...].astype(jnp.float32)

    dtype = k_ref.dtype

    k_ = k_ref[...].transpose(1, 0, 2)
    v_ = v_ref[...].transpose(1, 0, 2)

    log_f_cumsum_ = None
    if log_f_cumsum_ref is not None:
        log_f_cumsum_ = log_f_cumsum_ref[...]

        if f_diagonal:
            log_f_cumsum_ = log_f_cumsum_.transpose(1, 0, 2)
        else:
            log_f_cumsum_ = log_f_cumsum_.transpose(1, 0)

    batched_diag = fused_scan and f_diagonal and Gk == 1 and Gv == 1 and Gf == 1 and N % HEAD_GROUP == 0

    if fused_scan:
        BLOCK_SIZE_S = k_.shape[1]

        if batched_diag:
            causal_mask = jnp.tril(jnp.ones((HEAD_GROUP, BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32))
        else:
            causal_mask = jnp.tril(jnp.ones((BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32))

    if batched_diag:
        for n0 in range(0, N, HEAD_GROUP):
            g = slice(n0, n0 + HEAD_GROUP)

            log_f_g = log_f_cumsum_ref[...][:, g, :].transpose(1, 0, 2).astype(jnp.float32)
            log_f_g = jax.lax.dot_general(
                causal_mask, log_f_g, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
            )
            log_f_last_g = log_f_g[:, -1]  # (HEAD_GROUP, K)

            k_g = (k_[g].astype(dtype) * jnp.exp(log_f_last_g[:, None, :] - log_f_g)).astype(dtype)
            f_last_g = jnp.exp(log_f_last_g)

            for BLOCK_ID_V in range(NUM_BLOCKS_V):
                start = BLOCK_ID_V * BLOCK_SIZE_V
                end = start + BLOCK_SIZE_V

                v_g = v_[g][:, :, start:end].astype(dtype)
                h_g = h_scratch[g, :, start:end]
                h_ref[g, :, start:end] = h_g.astype(h_ref.dtype)

                h_g *= f_last_g[:, :, None]
                h_g += jax.lax.dot_general(k_g, v_g, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
                h_scratch[g, :, start:end] = h_g
    else:
        for n in range(N):
            k = k_[n // Gk].astype(dtype)

            if log_f_cumsum_ is not None:
                log_f = log_f_cumsum_[n // Gf].astype(jnp.float32)
                if fused_scan:
                    log_f = jax.lax.dot_general(causal_mask, log_f, (((1,), (0,)), ((), ())))
                log_f_last = log_f[-1]

                if f_diagonal:
                    f_last = jnp.exp(log_f_last[:, None])
                    k *= jnp.exp(log_f_last[None, :] - log_f)
                else:
                    f_last = jnp.exp(log_f_last)
                    k *= jnp.exp(log_f_last - log_f)[:, None]

                k = k.astype(dtype)

            for BLOCK_ID_V in range(NUM_BLOCKS_V):
                start = BLOCK_ID_V * BLOCK_SIZE_V
                end = start + BLOCK_SIZE_V

                v = v_[n // Gv][:, start:end].astype(dtype)
                h = h_scratch[n][:, start:end]
                h_ref[n, :, start:end] = h.astype(h_ref.dtype)

                if log_f_cumsum_ref is not None:
                    h *= f_last

                h += jax.lax.dot_general(k, v, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
                h_scratch[n, :, start:end] = h


@partial(jax.jit, static_argnames=("N", "BLOCK_SIZE_S", "BLOCK_SIZE_V", "fused_scan", "HEAD_GROUP"))
def _linear_attention_state_passing_core(
    k: jax.Array,
    v: jax.Array,
    log_f_cumsum: jax.Array | None,
    h0: jax.Array | None,
    N: int,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    fused_scan: bool = False,
    HEAD_GROUP: int = 8,
) -> jax.Array:
    B, S, Nk, K = k.shape
    Nv, V = v.shape[-2:]
    Nf = 0 if log_f_cumsum is None else log_f_cumsum.shape[2]

    Gk = N // Nk
    Gv = N // Nv
    Gf = None if log_f_cumsum is None else N // Nf

    f_diagonal = log_f_cumsum is not None and log_f_cumsum.ndim == 4
    if log_f_cumsum is not None:
        assert log_f_cumsum.shape == (B, S, Nf, K) if f_diagonal else (B, S, Nf)
        assert N % Nf == 0

    assert S % BLOCK_SIZE_S == 0

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    NUM_BLOCKS_V = ceil_divide(V, BLOCK_SIZE_V)
    assert (
        V == NUM_BLOCKS_V * BLOCK_SIZE_V
    ), "V must be an integer multiple of BLOCK_SIZE_V (host padding guarantees this)"

    f_spec = None
    if log_f_cumsum is not None:
        if f_diagonal:
            f_spec = pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nf, K),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            )
        else:
            f_spec = pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nf),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0),
            )

    kernel = pl.pallas_call(
        partial(
            _linear_attention_state_passing_kernel,
            N=N,
            Gk=Gk,
            Gv=Gv,
            Gf=Gf,
            f_diagonal=f_diagonal,
            NUM_BLOCKS_V=NUM_BLOCKS_V,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            fused_scan=fused_scan,
            HEAD_GROUP=HEAD_GROUP,
        ),
        out_shape=jax.ShapeDtypeStruct(shape=(B, NUM_BLOCKS_S * N, K, V), dtype=k.dtype),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nk, K),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nv, V),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            ),
            f_spec,
            (
                None
                if h0 is None
                else pl.BlockSpec(
                    block_shape=(None, N, K, V),
                    index_map=lambda BLOCK_ID_B, _: (BLOCK_ID_B, 0, 0, 0),
                )
            ),
        ),
        out_specs=pl.BlockSpec(
            block_shape=(None, N, K, V),
            index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
        ),
        scratch_shapes=[pltpu.VMEM((N, K, V), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, dimension_semantics=("parallel", "arbitrary")
        ),
    )

    return kernel(k, v, log_f_cumsum, h0)
