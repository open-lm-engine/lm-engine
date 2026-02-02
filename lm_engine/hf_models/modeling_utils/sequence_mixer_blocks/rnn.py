# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....utils import divide_if_divisible, is_xma_available
from ...cache import GenerationCache
from ...parameter import (
    mark_parameter_as_initialized,
    mark_parameter_as_mup_learning_rate,
    mark_parameter_as_no_weight_decay,
)
from ..activations import get_activation_function, is_glu
from ..convolution import ParameterizedConv1d
from ..linear import ParameterizedLinear
from ..normalization import get_normalization_function
from .causal_convolution import causal_convolution
from .utils import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence


if is_xma_available():
    from xma.layers.rnn import rnn


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_head_dim: int,
        output_size: int,
        num_input_heads: int,
        num_weight_heads: int,
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
    ) -> RNN:
        super().__init__()

        self.num_input_heads = num_input_heads
        self.num_weight_heads = num_weight_heads
        self.num_heads = max(num_input_heads, num_weight_heads)

        divide_if_divisible(self.num_heads, self.num_input_heads)
        divide_if_divisible(self.num_heads, self.num_weight_heads)

        self.gradient_clipping = gradient_clipping

        self.state_head_dim = state_head_dim
        self.state_size = self.num_heads * self.state_head_dim
        self.kernel_size = kernel_size
        self.activation_string = activation_function
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer

        self.x_shape = self.num_input_heads * self.state_head_dim
        self.g_shape = self.num_heads * self.state_head_dim

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.state_weight_std = std

        self.input_projection = ParameterizedLinear(input_size, self.x_shape + self.g_shape, bias=add_bias, std=std)

        if kernel_size is None:
            assert activation_function is None
        else:
            assert not is_glu(self.activation_string)

            self.conv1d = ParameterizedConv1d(
                in_channels=self.state_size,
                out_channels=self.state_size,
                kernel_size=kernel_size,
                bias=add_bias,
                padding=kernel_size - 1,
                groups=self.state_size,
                std=std,
            )

            mark_parameter_as_mup_learning_rate(self.conv1d.weight)

        self.activation_function = get_activation_function(self.activation_string)

        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.output_projection = ParameterizedLinear(self.state_size, output_size, bias=False, std=std)

        self.norm = get_normalization_function(normalization_function, self.state_size)

        mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        mark_parameter_as_mup_learning_rate(self.state_weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)

        mark_parameter_as_no_weight_decay(self.state_weight)

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
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

            B, S = x.size()[:2]

            if attention_mask is not None:
                cu_seqlens, max_seqlen = compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask)
                x = pack_sequence(inputs=x, cu_seqlens=cu_seqlens)

        c, h = (None, None) if cache_params is None else cache_params.get_cache(self.layer_idx)

        x = self.input_projection(x)
        x, g = x.split((self.x_shape, self.g_shape), dim=-1)

        if self.kernel_size is None:
            x = self.activation_function(x)
        else:
            x, c = causal_convolution(
                hidden_states=x,
                input_state=c,
                attention_mask=attention_mask,
                conv1d_weight=self.conv1d.weight,
                conv1d_bias=self.conv1d.bias,
                conv1d_num_groups=self.state_size,
                return_cache_state=cache_params is not None,
                activation_string=self.activation_string,
                conv1d_padding=self.kernel_size - 1,
                conv1d_stride=1,
            )

        x = x.view(*x.size()[:-1], -1, self.state_head_dim)

        x, h = rnn(
            input=x,
            weight=self.state_weight,
            input_state=h,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if not self.use_padding_free_transformer and attention_mask is not None:
            x = unpack_sequence(inputs=x, cu_seqlens=cu_seqlens, output_shape=(B, S, *x.size()[1:]))

        if cache_params is not None:
            cache_params.update(conv_state=c, ssm_state=h, num_tokens_added=x.size(1), layer_idx=self.layer_idx)

        x = x.flatten(-2, -1)
        x = x * F.silu(g)
        x = self.norm(x)
        x = self.output_projection(x)

        return x

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)
        mark_parameter_as_initialized(self.state_weight)

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping}\nweight_shape: {str(self.state_weight.shape)}"
