# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_fma_available, print_ranks_all
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import is_glu
from ..convolution import ParameterizedConv1d
from ..linear import ParameterizedLinear, ParameterizedLowRankLinear
from ..normalization import get_normalization_function
from .causal_convolution import causal_convolution
from .utils import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence


if is_fma_available():
    from fma import KernelBackend
    from fma.modules.msu import msu


class MSU(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int,
        low_rank: int | None,
        low_rank_norm: bool,
        num_heads: int,
        kernel_size: int | None,
        activation_function: str | None,
        add_bias: bool,
        gradient_clipping: float | None,
        initializer_range: float,
        m_width: float,
        init_method: str,
        normalization_function: str | None,
        num_layers: int,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> MSU:
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.low_rank = low_rank
        self.num_heads = num_heads
        self.low_rank_head_dim = divide_if_divisible(self.low_rank, self.num_heads, "")
        self.kernel_size = kernel_size
        self.activation_string = activation_function
        self.gradient_clipping = gradient_clipping
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer
        self.state_head_dim = divide_if_divisible(self.state_size, self.num_heads, "")
        self.expansion_factor = divide_if_divisible(self.state_size, self.low_rank, "")

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.state_weight_std = std

        self.reset_input_shape = self.num_heads
        self.C_shape = self.num_heads
        self.gate_shape = self.low_rank
        self.input_shape = self.low_rank
        self.forget_input_shape = self.low_rank

        self.conv_dim = (
            self.input_shape + self.forget_input_shape + self.reset_input_shape + self.C_shape + self.expansion_factor
        )

        self.input_projection = ParameterizedLinear(
            self.input_size, self.conv_dim + self.gate_shape, bias=add_bias, std=std
        )

        if kernel_size is None:
            assert activation_function is None
        else:
            assert not is_glu(self.activation_string)

            self.conv1d = ParameterizedConv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                kernel_size=kernel_size,
                bias=add_bias,
                padding=kernel_size - 1,
                groups=self.conv_dim,
                std=std,
            )

        self.state_weight = nn.Parameter(torch.empty(3 * self.num_heads, self.state_head_dim, self.state_head_dim))

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.up_projection = ParameterizedLinear(self.state_head_dim, self.low_rank_head_dim, bias=False, std=std)
        self.output_projection = ParameterizedLinear(self.low_rank, self.output_size, bias=False, std=std)

        self.norm = get_normalization_function(normalization_function, self.gate_shape)
        self.input_norm = get_normalization_function("rmsnorm", self.input_shape)
        self.forget_norm = get_normalization_function("rmsnorm", self.forget_input_shape)
        self.reset_norm = get_normalization_function("rmsnorm", self.reset_input_shape)

        self.residual_weight = nn.Parameter(torch.empty(self.num_heads))

        self.state_weight_norm = get_normalization_function(
            "p_norm", self.state_head_dim * self.state_head_dim, elementwise_affine=False, p=2
        )

        mark_parameter_as_mup_learning_rate(self.conv1d.weight)
        mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        mark_parameter_as_mup_learning_rate(self.state_weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)
        mark_parameter_as_mup_learning_rate(self.up_projection.weight)
        mark_parameter_as_mup_learning_rate(self.residual_weight)

        mark_parameter_as_no_weight_decay(self.state_weight)
        mark_parameter_as_no_weight_decay(self.residual_weight)

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if self.use_padding_free_transformer:
            assert cache_params is None
            assert attention_mask is None
        else:
            assert cu_seqlens is None
            assert max_seqlen is None

            B, S = input.size()[:2]

            if attention_mask is not None:
                cu_seqlens, max_seqlen = compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask)
                input = pack_sequence(inputs=input, cu_seqlens=cu_seqlens)

        input_state = None if cache_params is None else cache_params.get_cache(self.layer_idx)
        conv_state = None

        input = self.input_projection(input)
        input, gate = input.split((self.conv_dim, self.gate_shape), dim=-1)

        if self.kernel_size is not None:
            input, conv_state = causal_convolution(
                hidden_states=input,
                input_state=conv_state,
                attention_mask=attention_mask,
                conv1d_weight=self.conv1d.weight,
                conv1d_bias=self.conv1d.bias,
                conv1d_num_groups=self.conv_dim,
                return_cache_state=cache_params is not None,
                activation_string=self.activation_string,
                conv1d_padding=self.kernel_size - 1,
                conv1d_stride=1,
            )

        input, forget_input, reset_input, C, expander = input.split(
            (self.input_shape, self.forget_input_shape, self.reset_input_shape, self.C_shape, self.expansion_factor),
            dim=-1,
        )

        input = self.input_norm(input)
        forget_input = self.forget_norm(forget_input)
        reset_input = self.reset_norm(reset_input)

        # LR = N * LRH
        # SS = N * SH

        # B, S, N, 1, LRH
        input, forget_input = [
            i.view(*input.size()[:-1], self.num_heads, self.low_rank_head_dim).unsqueeze(-2)
            for i in (input, forget_input)
        ]

        # B, S, N, LRH, 1
        expander = expander.unsqueeze(-2).expand(-1, -1, self.num_heads, -1).unsqueeze(-1)

        input = input * expander
        forget_input = forget_input * expander

        input = input.flatten(-2, -1)
        forget_input = forget_input.flatten(-2, -1)

        residual = self.residual_weight.unsqueeze(-1).unsqueeze(0).unsqueeze(0) * input

        state_weight = self.state_weight_norm(self.state_weight.view(3 * self.num_heads, -1)).view_as(
            self.state_weight
        )
        weight, forget_weight, reset_weight = state_weight.chunk(3, dim=0)

        input = msu(
            input=input,
            weight=weight,
            forget_input=forget_input,
            forget_weight=forget_weight,
            reset_input=reset_input,
            reset_weight=reset_weight,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.triton if is_kernel_allowed(Kernel.gru) else KernelBackend.torch,
        )

        if not self.use_padding_free_transformer and attention_mask is not None:
            input = unpack_sequence(inputs=input, cu_seqlens=cu_seqlens, output_shape=(B, S, *input.size()[1:]))

        if cache_params is not None:
            cache_params.update(state=input[:, -1], num_tokens_added=input.size(1), layer_idx=self.layer_idx)

        input = residual + input * C.unsqueeze(-1)

        input = self.up_projection(input)
        input = input.view(*input.size()[:-2], -1)

        input = input * F.silu(gate)
        input = self.norm(input)

        input = self.output_projection(input)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)
        nn.init.ones_(self.residual_weight)

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping}\nweight_shape: {str(self.state_weight.shape)}"
