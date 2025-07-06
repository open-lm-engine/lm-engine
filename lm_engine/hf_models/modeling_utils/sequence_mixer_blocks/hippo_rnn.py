# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_cute_kernels_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import is_glu
from ..convolution import ParameterizedConv1d
from ..linear import ParameterizedLinear
from ..normalization import get_normalization_function
from .causal_convolution import causal_convolution
from .packing import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence


if is_cute_kernels_available():
    from cute_kernels import KernelBackend
    from cute_kernels.modules.hippo_rnn import hippo_rnn_cute


class HiPPO_RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int,
        num_heads: int,
        num_groups: int | None,
        kernel_size: int | None,
        activation_function: str | None,
        hippo_size: int,
        hippo_measure: str,
        add_bias: bool,
        gradient_clipping: float | None,
        initializer_range: float,
        m_width: float,
        init_method: str,
        normalization_function: str | None,
        num_layers: int,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> HiPPO_RNN:
        super().__init__()

        assert hippo_measure == "legs"

        self.hippo_measure = hippo_measure
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.activation_string = activation_function
        self.gradient_clipping = gradient_clipping
        self.hippo_size = hippo_size
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer
        self.state_head_dim = divide_if_divisible(self.state_size, self.num_heads, "")
        self.is_gated_normalization = normalization_function == "silu_gated_rmsnorm"

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.state_weight_std = std

        if kernel_size is None:
            assert num_groups is None
            assert activation_function is None
        else:
            divide_if_divisible(input_size, num_groups, "")

            self.conv1d = ParameterizedConv1d(
                in_channels=input_size,
                out_channels=input_size * 2 if is_glu(self.activation_string) else input_size,
                kernel_size=kernel_size,
                bias=add_bias,
                padding=kernel_size - 1,
                groups=num_groups,
                std=std,
            )

        self.input_projection = ParameterizedLinear(
            self.input_size,
            self.state_size + (self.state_size if self.is_gated_normalization else 0),
            bias=add_bias,
            std=std,
        )

        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))
        self.hippo_weight = nn.Parameter(torch.empty(self.num_heads, self.hippo_size, self.state_head_dim))
        self.compress_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim))

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.output_projection = ParameterizedLinear(self.state_size, self.output_size, bias=False, std=std)

        self.register_buffer("A", torch.empty(hippo_size, hippo_size))
        self.register_buffer("B", torch.empty(hippo_size))

        self.norm = get_normalization_function(normalization_function, self.state_size)
        self.input_norm = get_normalization_function("rmsnorm", self.state_size)

        self.state_weight_norm = get_normalization_function(
            "p_norm", self.state_head_dim * self.state_head_dim, elementwise_affine=False, p=2
        )
        self.hippo_weight_norm = get_normalization_function(
            "p_norm", self.hippo_size * self.state_head_dim, elementwise_affine=False, p=2
        )

        mark_parameter_as_mup_learning_rate(self.conv1d.weight)
        mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        mark_parameter_as_mup_learning_rate(self.state_weight)
        mark_parameter_as_mup_learning_rate(self.hippo_weight)
        mark_parameter_as_mup_learning_rate(self.compress_weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)

        mark_parameter_as_no_weight_decay(self.state_weight)
        mark_parameter_as_no_weight_decay(self.hippo_weight)

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

        input_state, hippo_state = (None, None) if cache_params is None else cache_params.get_cache(self.layer_idx)
        conv_state = None

        if self.kernel_size is not None:
            input, conv_state = causal_convolution(
                hidden_states=input,
                input_state=conv_state,
                attention_mask=attention_mask,
                conv1d_weight=self.conv1d.weight,
                conv1d_bias=self.conv1d.bias,
                conv1d_num_groups=self.num_groups,
                return_cache_state=cache_params is not None,
                activation_string=self.activation_string,
                conv1d_padding=self.kernel_size - 1,
                conv1d_stride=1,
            )

        input = self.input_projection(input)

        if self.is_gated_normalization:
            input, gate = input.chunk(2, dim=-1)

        input = self.input_norm(input)
        input = input.view(*input.size()[:-1], self.num_heads, self.state_head_dim)

        state_weight = self.state_weight_norm(self.state_weight.view(self.num_heads, -1)).view_as(self.state_weight)
        hippo_weight = self.hippo_weight_norm(self.hippo_weight.view(self.num_heads, -1)).view_as(self.hippo_weight)

        input = hippo_rnn_cute(
            input=input,
            weight=state_weight,
            hippo_weight=hippo_weight,
            compress_weight=self.compress_weight,
            hippo_A=self.A.type_as(input),
            hippo_B=self.B.type_as(input),
            input_state=input_state,
            hippo_state=hippo_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.triton if is_kernel_allowed(Kernel.hippo_rnn_cute) else KernelBackend.torch,
        )

        if not self.use_padding_free_transformer and attention_mask is not None:
            input = unpack_sequence(inputs=input, cu_seqlens=cu_seqlens, desired_shape=(B, S, *input.size()[1:]))

        if cache_params is not None:
            cache_params.update(state=input[:, -1], num_tokens_added=input.size(1), layer_idx=self.layer_idx)

        input = input.view(*input.size()[:-2], -1)

        if self.is_gated_normalization:
            input = self.norm(input, gate)
        else:
            input = self.norm(input)

        input = self.output_projection(input)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)
        nn.init.normal_(self.hippo_weight, std=self.state_weight_std)
        nn.init.normal_(self.compress_weight, std=self.state_weight_std)

        arange = np.arange(self.hippo_size, dtype=np.float64)
        r = 2 * arange + 1

        B = np.sqrt(r)

        M = np.tril(r) - np.diag(arange)
        T = B[:, None]
        T_inv = 1 / B[None, :]
        r = 2 * arange + 1

        A = T * T_inv * M

        A = torch.tensor(A)
        B = torch.tensor(B)

        self.A.copy_(A)
        self.B.copy_(B)

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping}\nweight_shape = {str(self.state_weight.shape)}\nmeasure = {self.hippo_measure}"
