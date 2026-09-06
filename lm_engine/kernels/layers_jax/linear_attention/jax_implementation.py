# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import jax
import jax.numpy as jnp


def _linear_attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f: jax.Array | None,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
) -> tuple[jax.Array, jax.Array | None]:
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]
    Nf = 0 if log_f is None else log_f.shape[2]

    if log_f is not None:
        if log_f.ndim == 3:
            log_f = log_f[..., None]

        log_f = log_f.astype(jnp.float32)

    N = max(Nq, Nk, Nv, Nf)
    dtype = q.dtype

    q = jnp.repeat(q, N // Nq, axis=-2)
    k = jnp.repeat(k, N // Nk, axis=-2)
    v = jnp.repeat(v, N // Nv, axis=-2)
    if log_f is not None:
        log_f = jnp.repeat(log_f, N // Nf, axis=-2)

    h = jnp.zeros((B, N, K, V), dtype=jnp.float32) if h0 is None else h0.astype(jnp.float32)

    y = []
    for s in range(S):
        if log_f is not None:
            h *= jnp.exp(log_f[:, s, ..., None].astype(jnp.float32))

        h += k[:, s, ..., None].astype(jnp.float32) * v[:, s, :, None, :].astype(jnp.float32)
        y.append(jnp.einsum("bnk,bnkv->bnv", q[:, s].astype(jnp.float32), h))

    y = jnp.stack(y, axis=1) * attention_multiplier

    if not output_state:
        h = None

    return y.astype(dtype), h
