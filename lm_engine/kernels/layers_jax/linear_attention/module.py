# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from ...accelerator import Accelerator, KernelBackend
from ...math import divide_if_divisible
from ..depthwise_causal_convolution import DepthwiseCausalConvolutionJAX
from .op import linear_attention_jax


class _Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array | None

    @staticmethod
    def init(in_size: int, out_size: int, use_bias: bool, *, key: PRNGKeyArray) -> _Linear:
        weight_key, bias_key = jax.random.split(key, 2)
        bound = in_size**-0.5

        weight = jax.random.uniform(weight_key, (in_size, out_size), minval=-bound, maxval=bound)
        bias = jax.random.uniform(bias_key, (out_size,), minval=-bound, maxval=bound) if use_bias else None

        return _Linear(weight=weight, bias=bias)

    def __call__(self, input: jax.Array) -> jax.Array:
        output = input @ self.weight
        if self.bias is not None:
            output = output + self.bias

        return output


class LinearAttentionJAX(eqx.Module):
    input_projection: _Linear
    output_projection: _Linear
    conv1d: DepthwiseCausalConvolutionJAX | None

    input_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    num_query_heads: int = eqx.field(static=True)
    num_key_heads: int = eqx.field(static=True)
    num_value_heads: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    key_head_dim: int = eqx.field(static=True)
    value_head_dim: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)

    attention_multiplier: float | None = eqx.field(static=True)
    BLOCK_SIZE_S: int = eqx.field(static=True)
    BLOCK_SIZE_V: int = eqx.field(static=True)

    @staticmethod
    def init(
        input_size: int,
        output_size: int,
        key_head_dim: int,
        value_head_dim: int,
        num_query_heads: int,
        num_key_heads: int,
        num_value_heads: int,
        add_bias: bool,
        *,
        attention_multiplier: float | None = None,
        BLOCK_SIZE_S: int = 256,
        BLOCK_SIZE_V: int = 128,
        kernel_size: int | None = None,
        conv_activation_function: str | Callable[[jax.Array], jax.Array] | None = None,
        key: PRNGKeyArray,
    ) -> LinearAttentionJAX:
        num_heads = max(num_query_heads, num_key_heads, num_value_heads)

        divide_if_divisible(num_heads, num_query_heads)
        divide_if_divisible(num_heads, num_key_heads)
        divide_if_divisible(num_heads, num_value_heads)

        lane_count = Accelerator.get_lane_count()

        if BLOCK_SIZE_V <= 0 or BLOCK_SIZE_V % lane_count != 0:
            raise ValueError(
                f"BLOCK_SIZE_V ({BLOCK_SIZE_V}) must be a positive multiple of {lane_count} "
                "(pallas-kernel envelope; enforced at init so a later kernel_backend='pallas' call cannot fail mid-run)"
            )

        query_size = num_query_heads * key_head_dim
        key_size = num_key_heads * key_head_dim
        value_size = num_value_heads * value_head_dim
        qkv_size = query_size + key_size + value_size
        heads_value_size = num_heads * value_head_dim

        key_input_projection, key_output_projection, key_conv = jax.random.split(key, 3)

        input_projection = _Linear.init(input_size, qkv_size, use_bias=add_bias, key=key_input_projection)
        output_projection = _Linear.init(heads_value_size, output_size, use_bias=add_bias, key=key_output_projection)

        conv1d = (
            None
            if kernel_size is None
            else DepthwiseCausalConvolutionJAX.init(
                qkv_size,
                kernel_size=kernel_size,
                activation_function=conv_activation_function,
                add_bias=add_bias,
                key=key_conv,
            )
        )

        return LinearAttentionJAX(
            input_projection=input_projection,
            output_projection=output_projection,
            conv1d=conv1d,
            input_size=input_size,
            output_size=output_size,
            num_query_heads=num_query_heads,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            num_heads=num_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            state_size=key_head_dim * value_head_dim,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
        )

    def __call__(
        self,
        input: jax.Array,
        input_state: jax.Array | None = None,
        conv_state: jax.Array | None = None,
        output_conv_state: bool = False,
        output_state: bool = True,
        *,
        kernel_backend: KernelBackend | None = None,
    ) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
        B, S, _ = input.shape

        query_size = self.num_query_heads * self.key_head_dim
        key_size = self.num_key_heads * self.key_head_dim

        projected = self.input_projection(input)

        if self.conv1d is None:
            assert conv_state is None
        else:
            projected, conv_state = self.conv1d(
                projected, input_state=conv_state, output_state=output_conv_state, kernel_backend=kernel_backend
            )

        query, key, value = jnp.split(projected, [query_size, query_size + key_size], axis=-1)

        query = query.reshape(B, S, self.num_query_heads, self.key_head_dim)
        key = key.reshape(B, S, self.num_key_heads, self.key_head_dim)
        value = value.reshape(B, S, self.num_value_heads, self.value_head_dim)

        input, input_state = linear_attention_jax(
            query=query,
            key=key,
            value=value,
            input_state=input_state,
            attention_multiplier=self.attention_multiplier,
            output_state=output_state,
            BLOCK_SIZE_S=self.BLOCK_SIZE_S,
            BLOCK_SIZE_V=self.BLOCK_SIZE_V,
            kernel_backend=kernel_backend,
        )

        input = input.reshape(B, S, self.num_heads * self.value_head_dim)
        input = self.output_projection(input)

        return input, input_state, conv_state
