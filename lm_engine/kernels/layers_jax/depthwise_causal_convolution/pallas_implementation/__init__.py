# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.numpy as jnp

from ....math import ceil_divide
from .backward import _backward_core
from .forward import _forward_core


def _get_block_size_s(H: int, dtype=jnp.float32) -> int:
    # the kernel stack holds several (BLOCK_SIZE_S, dtype) tiles and vmem overflows once
    # BLOCK_SIZE_C * H * bytes_per_elem exceeds 2**21 bytes for bf16 / 2**20 bytes for fp32
    # (measured), so shrink the block for wide H / wider dtypes. Narrow H wants the block as
    # big as possible instead: more rows per program means fewer HBM round-trips, which is
    # what sets the kernel's bandwidth at small hidden sizes
    budget = (1 << 21) if jnp.dtype(dtype).itemsize == 2 else (1 << 20)
    block_size = 1 << (budget // (H * jnp.dtype(dtype).itemsize)).bit_length() - 1
    return min(1024, max(8, block_size))


def _pad_h0(h0: jax.Array, K: int) -> jax.Array:
    state_size = K - 1
    pad = ceil_divide(state_size, 8) * 8
    return jnp.pad(h0, ((0, 0), (pad - state_size, 0), (0, 0)))


def _forward_run(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h0: jax.Array | None,
    output_state: bool,
    ACTIVATION: str | None,
) -> tuple[jax.Array, jax.Array | None, jax.Array]:
    W = jnp.transpose(W, (1, 0))
    b = None if b is None else b[None, :]

    if h0 is not None:
        h0 = jnp.transpose(h0, (0, 2, 1)).astype(x.dtype)
        h0 = _pad_h0(h0, K=W.shape[0])

    # the per-block input states are saved from the forward as vjp residuals (a small
    # (B, ceil(S / BLOCK_SIZE_S), PAD, H) tensor) so the backward can consume them directly
    # instead of re-deriving them with an extra pass over the input
    y, h = _forward_core(
        x=x, W=W, b=b, h0=h0, BLOCK_SIZE_S=_get_block_size_s(x.shape[-1], x.dtype), ACTIVATION=ACTIVATION
    )

    if output_state:
        state_size = W.shape[0] - 1
        if h0 is None:
            ht = (
                jnp.pad(x, ((0, 0), (state_size - x.shape[1], 0), (0, 0)))
                if x.shape[1] < state_size
                else x[:, -state_size:, :]
            )
        else:
            ht = jnp.concatenate([h0.astype(x.dtype), x], axis=1)[:, -state_size:, :]

        ht = jnp.transpose(ht, (0, 2, 1))
    else:
        ht = None

    return y, ht, h


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _depthwise_causal_convolution_pallas(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h0: jax.Array | None,
    output_state: bool,
    ACTIVATION: str | None,
) -> tuple[jax.Array, jax.Array | None]:
    y, ht, _ = _forward_run(x=x, W=W, b=b, h0=h0, output_state=output_state, ACTIVATION=ACTIVATION)
    return y, ht


def _depthwise_causal_convolution_forward(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h0: jax.Array | None,
    output_state: bool,
    ACTIVATION: str | None,
) -> tuple[tuple[jax.Array, jax.Array | None], tuple]:
    y, ht, h = _forward_run(x=x, W=W, b=b, h0=h0, output_state=output_state, ACTIVATION=ACTIVATION)
    return (y, ht), (x, W, b, h0, h)


def _depthwise_causal_convolution_backward(
    output_state: bool, ACTIVATION: str | None, residuals: tuple, cotangents: tuple
) -> tuple:
    x, W, b, h0, h_states = residuals
    dy, dht = cotangents

    K = W.shape[-1]
    W = jnp.transpose(W, (1, 0))

    dht = None if dht is None or not output_state else jnp.transpose(dht, (0, 2, 1))

    dx, dW, db, dh0 = _backward_core(
        x=x,
        W=W,
        b=None if b is None else b[None, :],
        h=h_states,
        dy=dy,
        dht=dht,
        BLOCK_SIZE_S=_get_block_size_s(x.shape[-1], x.dtype),
        K=K,
        ACTIVATION=ACTIVATION,
    )

    dW = jnp.transpose(dW, (1, 0))
    db = None if b is None else db[0]
    dh0 = None if h0 is None else jnp.transpose(dh0[:, 1 - K :, :], (0, 2, 1))

    return dx, dW, db, dh0


_depthwise_causal_convolution_pallas.defvjp(
    _depthwise_causal_convolution_forward, _depthwise_causal_convolution_backward
)
