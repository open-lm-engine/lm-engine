# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn

from ...accelerator import KernelBackend
from ...math import divide_if_divisible
from ..depthwise_causal_convolution import DepthwiseCausalConvolution
from .op import m2rnn


class M2RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        key_head_dim: int,
        value_head_dim: int,
        output_size: int,
        num_query_heads: int,
        num_key_heads: int,
        num_value_heads: int,
        num_forget_input_heads: int,
        num_weight_heads: int,
        add_bias: bool,
        gradient_clipping: float | None,
        kernel_size: int | None = None,
        conv_activation_function: str | None = None,
    ) -> None:
        super().__init__()

        self.gradient_clipping = gradient_clipping
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim

        self.num_query_heads = num_query_heads
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.num_forget_input_heads = num_forget_input_heads
        self.num_weight_heads = num_weight_heads

        self.num_heads = max(num_query_heads, num_key_heads, num_value_heads, num_forget_input_heads, num_weight_heads)
        self.state_size = self.num_heads * self.value_head_dim

        divide_if_divisible(self.num_heads, self.num_query_heads)
        divide_if_divisible(self.num_heads, self.num_key_heads)
        divide_if_divisible(self.num_heads, self.num_value_heads)
        divide_if_divisible(self.num_heads, self.num_forget_input_heads)
        divide_if_divisible(self.num_heads, self.num_weight_heads)

        self.q_shape = self.num_query_heads * self.key_head_dim
        self.k_shape = self.num_key_heads * self.key_head_dim
        self.v_shape = self.num_value_heads * self.value_head_dim
        self.x_shape = self.q_shape + self.k_shape + self.v_shape

        self.input_projection = nn.Linear(input_size, self.x_shape + self.num_forget_input_heads, bias=add_bias)

        self.state_weight = nn.Parameter(torch.empty(self.num_weight_heads, self.value_head_dim, self.value_head_dim))
        self.output_projection = nn.Linear(self.state_size, output_size, bias=False)

        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.conv1d = DepthwiseCausalConvolution(
                hidden_size=self.x_shape,
                kernel_size=kernel_size,
                activation_function=conv_activation_function,
                add_bias=add_bias,
            )

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        input_state: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        output_conv_state: bool = False,
        *,
        kernel_backend: KernelBackend | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        input = self.input_projection(input)
        input, f = input.split((self.x_shape, self.num_forget_input_heads), dim=-1)

        if self.kernel_size is None:
            assert conv_state is None
        else:
            assert cu_seqlens is None, "depthwise causal conv does not support cu_seqlens"
            input, conv_state = self.conv1d(input, input_state=conv_state, output_state=output_conv_state)

        q, k, v = input.split((self.q_shape, self.k_shape, self.v_shape), dim=-1)

        q = q.view(*q.size()[:-1], -1, self.key_head_dim)
        k = k.view(*k.size()[:-1], -1, self.key_head_dim)
        v = v.view(*v.size()[:-1], -1, self.value_head_dim)

        input, input_state = m2rnn(
            query=q,
            key=k,
            value=v,
            weight=self.state_weight,
            forget_input=f,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=kernel_backend,
        )

        input = input.flatten(-2, -1)
        input = self.output_projection(input)

        return input, input_state, conv_state

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight)
