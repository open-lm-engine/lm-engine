# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from ...accelerator import KernelBackend
from .op import depthwise_causal_convolution_jax


class DepthwiseCausalConvolutionJAX(eqx.Module):
    weight: jax.Array
    bias: jax.Array | None

    hidden_size: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    activation_function: str | None = eqx.field(static=True)

    @staticmethod
    def init(
        hidden_size: int,
        kernel_size: int,
        activation_function: str | None,
        add_bias: bool,
        *,
        key: PRNGKeyArray,
    ) -> DepthwiseCausalConvolutionJAX:
        assert kernel_size > 1

        weight_key, bias_key = jax.random.split(key, 2)
        bound = kernel_size**-0.5

        weight = jax.random.uniform(weight_key, (hidden_size, kernel_size), minval=-bound, maxval=bound)
        bias = jax.random.uniform(bias_key, (hidden_size,), minval=-bound, maxval=bound) if add_bias else None

        return DepthwiseCausalConvolutionJAX(
            weight=weight,
            bias=bias,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            state_size=kernel_size - 1,
            activation_function=activation_function,
        )

    def __call__(
        self,
        input: jax.Array,
        input_state: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        output_state: bool = False,
        *,
        kernel_backend: KernelBackend | None = None,
    ) -> tuple[jax.Array, jax.Array | None]:
        return depthwise_causal_convolution_jax(
            input=input,
            weight=self.weight,
            bias=self.bias,
            input_state=input_state,
            attention_mask=attention_mask,
            output_state=output_state,
            activation_function=self.activation_function,
            kernel_backend=kernel_backend,
        )
