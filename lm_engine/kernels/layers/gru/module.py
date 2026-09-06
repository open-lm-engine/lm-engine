# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...accelerator import KernelBackend
from ...math import divide_if_divisible
from ..depthwise_causal_convolution import DepthwiseCausalConvolution
from .op import gru


class GRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_head_dim: int,
        output_size: int,
        num_input_heads: int,
        num_forget_input_heads: int,
        num_reset_input_heads: int,
        num_weight_heads: int,
        num_forget_weight_heads: int,
        num_reset_weight_heads: int,
        add_bias: bool,
        gradient_clipping: float | None,
        kernel_size: int | None = None,
        conv_activation_function: str | None = None,
    ) -> GRU:
        super().__init__()

        self.num_input_heads = num_input_heads
        self.num_forget_input_heads = num_forget_input_heads
        self.num_reset_input_heads = num_reset_input_heads
        self.num_weight_heads = num_weight_heads
        self.num_forget_weight_heads = num_forget_weight_heads
        self.num_reset_weight_heads = num_forget_weight_heads

        self.num_heads = max(
            num_input_heads,
            num_forget_input_heads,
            num_reset_input_heads,
            num_weight_heads,
            num_forget_weight_heads,
            num_reset_weight_heads,
        )

        divide_if_divisible(self.num_heads, self.num_input_heads)
        divide_if_divisible(self.num_heads, self.num_forget_input_heads)
        divide_if_divisible(self.num_heads, self.num_reset_input_heads)

        divide_if_divisible(self.num_heads, self.num_weight_heads)
        divide_if_divisible(self.num_heads, self.num_forget_weight_heads)
        divide_if_divisible(self.num_heads, self.num_reset_weight_heads)

        self.gradient_clipping = gradient_clipping
        self.state_head_dim = state_head_dim
        self.state_size = self.num_heads * self.state_head_dim
        self.x_shape = self.num_input_heads * self.state_head_dim
        self.xf_shape = self.num_forget_input_heads * self.state_head_dim
        self.xr_shape = self.num_reset_input_heads * self.state_head_dim

        self.input_projection = nn.Linear(input_size, self.x_shape + self.xf_shape + self.xr_shape, bias=add_bias)

        self.state_weight = nn.Parameter(torch.empty(self.num_weight_heads, self.state_head_dim, self.state_head_dim))
        self.forget_weight = nn.Parameter(
            torch.empty(self.num_forget_weight_heads, self.state_head_dim, self.state_head_dim)
        )
        self.reset_weight = nn.Parameter(
            torch.empty(self.num_reset_weight_heads, self.state_head_dim, self.state_head_dim)
        )

        self.output_projection = nn.Linear(self.state_size, output_size, bias=add_bias)

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

        input, forget_gate, reset_gate = input.split((self.x_shape, self.xf_shape, self.xr_shape), dim=-1)

        if self.kernel_size is None:
            assert conv_state is None
        else:
            assert cu_seqlens is None, "depthwise causal conv does not support cu_seqlens"
            input, conv_state = self.conv1d(input, input_state=conv_state, output_state=output_conv_state)

        input, forget_gate, reset_gate = [
            i.view(*i.size()[:-1], -1, self.state_head_dim) for i in (input, forget_gate, reset_gate)
        ]

        input, input_state = gru(
            input=input,
            weight=self.state_weight,
            forget_input=forget_gate,
            forget_weight=self.forget_weight,
            reset_input=reset_gate,
            reset_weight=self.reset_weight,
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

    def extra_repr(self) -> str:
        output = super().extra_repr()
        return f"{output}\nstate size = {self.state_size} elements"
