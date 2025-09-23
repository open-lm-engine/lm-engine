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
from ....utils import divide_if_divisible, is_fma_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import is_glu
from ..convolution import ParameterizedConv1d
from ..linear import ParameterizedLinear
from ..normalization import get_normalization_function
from .causal_convolution import causal_convolution
from .utils import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence


if is_fma_available():
    from fma import KernelBackend
    from fma.modules.fru import fru


class FRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int,
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
    ) -> FRU:
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.activation_string = activation_function
        self.gradient_clipping = gradient_clipping
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer
        self.state_head_dim = divide_if_divisible(self.state_size, self.num_heads, "")

        self.num_groups = 1
        assert self.num_heads % self.num_groups == 0
        self.num_q_heads = self.num_heads
        self.num_k_heads = self.num_heads
        self.num_v_heads = self.num_heads
        self.num_f_heads = self.num_heads

        self.qk_head_dim = self.state_head_dim
        self.v_head_dim = self.state_head_dim

        self.q_shape = self.num_q_heads * self.qk_head_dim
        self.k_shape = self.num_k_heads * self.qk_head_dim
        self.f_shape = self.num_f_heads * self.qk_head_dim
        self.v_shape = self.num_v_heads * self.v_head_dim
        self.g_shape = self.num_v_heads * self.v_head_dim

        self.conv_dim = self.q_shape + self.k_shape + self.v_shape + self.f_shape

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.state_weight_std = std

        self.input_projection = ParameterizedLinear(
            self.input_size, self.conv_dim + self.g_shape, bias=add_bias, std=std
        )
        self.input_activation = nn.SiLU()

        if kernel_size is None:
            assert activation_function is None
        else:
            assert self.activation_string is None or not is_glu(self.activation_string)

            self.conv1d = ParameterizedConv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                kernel_size=kernel_size,
                bias=add_bias,
                padding=kernel_size - 1,
                groups=self.conv_dim,
                std=std,
            )

        self.logistic_factor = nn.Parameter(torch.empty(self.num_v_heads))
        mark_parameter_as_no_weight_decay(self.logistic_factor)

        self.state_weight = nn.Parameter(torch.empty(self.num_v_heads, self.state_head_dim, self.state_head_dim))

        self.forget_multiplier = nn.Parameter(torch.empty(self.num_f_heads))
        mark_parameter_as_no_weight_decay(self.forget_multiplier)

        self.forget_bias = nn.Parameter(torch.empty(self.num_f_heads, self.qk_head_dim))
        mark_parameter_as_no_weight_decay(self.forget_bias)

        self.log_activation_range = nn.Parameter(torch.empty((self.num_v_heads,)))
        mark_parameter_as_no_weight_decay(self.log_activation_range)

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.output_projection = ParameterizedLinear(self.g_shape, self.output_size, bias=False, std=std)

        self.norm = get_normalization_function("p_norm", self.state_head_dim, p=2, elementwise_affine=False)
        self.g_norm = get_normalization_function(normalization_function, self.state_size)

        mark_parameter_as_mup_learning_rate(self.conv1d.weight)
        mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        mark_parameter_as_mup_learning_rate(self.state_weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)

        # mark_parameter_as_no_weight_decay(self.state_weight)

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
        input, gate = input.split((self.conv_dim, self.g_shape), dim=-1)

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

        q, k, v, f = input.split((self.q_shape, self.k_shape, self.v_shape, self.f_shape), dim=-1)

        q = q.view(*q.size()[:-1], self.num_q_heads, self.qk_head_dim)
        k = k.view(*k.size()[:-1], self.num_k_heads, self.qk_head_dim)
        v = v.view(*v.size()[:-1], self.num_v_heads, self.v_head_dim)
        f = f.view(*f.size()[:-1], self.num_f_heads, self.qk_head_dim)

        k = self.norm(k)
        k = k.repeat(1, 1, self.num_v_heads // self.num_k_heads, 1)
        kvT = k[..., :, None] * v[..., None, :]

        f = (2 * torch.sigmoid(self.forget_multiplier[..., None])) * (f + self.forget_bias)
        f = f[..., None].expand_as(kvT)

        kvT = kvT.permute(0, 3, 1, 2, 4).flatten(0, 1)
        f = f.permute(0, 3, 1, 2, 4).flatten(0, 1)

        input = fru(
            input=kvT,
            weight=self.state_weight,
            forget_input=f,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            do_sigmoid_forget=True,
            kernel_backend=KernelBackend.triton if is_kernel_allowed(Kernel.gru) else KernelBackend.torch,
        )

        # FIXME
        # if not self.use_padding_free_transformer and attention_mask is not None:
        #     input = unpack_sequence(inputs=input, cu_seqlens=cu_seqlens, output_shape=(B, S, *input.size()[1:]))

        # if cache_params is not None:
        #     cache_params.update(state=input[:, -1], num_tokens_added=input.size(1), layer_idx=self.layer_idx)

        input = input.view(B, self.qk_head_dim, S, self.num_v_heads, self.v_head_dim).permute(0, 2, 3, 1, 4)
        input = input.view(B, S, self.num_v_heads, self.qk_head_dim, self.v_head_dim)
        input = q.unsqueeze(-2) @ input
        input = input.squeeze(-2).flatten(-2, -1)
        input = input * F.silu(gate)
        input = self.g_norm(input)

        input = self.output_projection(input)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)
        nn.init.zeros_(self.forget_multiplier)
        nn.init.zeros_(self.forget_bias)
        nn.init.zeros_(self.logistic_factor)
        nn.init.zeros_(self.log_activation_range)

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping}\nweight_shape: {str(self.state_weight.shape)}"
