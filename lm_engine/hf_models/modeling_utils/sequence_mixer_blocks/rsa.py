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
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import is_glu
from ..convolution import ParameterizedConv1d
from ..decay_gate import SoftplusDecayGate
from ..linear import ParameterizedLinear
from ..normalization import get_normalization_function
from .causal_convolution import causal_convolution
from .utils import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence


if is_xma_available():
    from xma import rsa


class RSA(nn.Module):
    def __init__(
        self,
        input_size: int,
        k_head_dim: int,
        v_head_dim: int,
        output_size: int,
        num_q_heads: int,
        num_k_heads: int,
        num_v_heads: int,
        num_f_heads: int,
        num_g_heads: int,
        num_weight_heads: int,
        use_residual: bool,
        kernel_size: int | None,
        activation_function: str | None,
        add_bias: bool,
        gradient_clipping: float | None,
        initializer_range: float,
        m_width: float,
        init_method: str,
        normalization_function: str | None,
        use_softplus_decay: bool,
        norm_after_flatten: bool,
        num_layers: int,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> RSA:
        super().__init__()

        self.input_size = input_size
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.activation_string = activation_function
        self.gradient_clipping = gradient_clipping
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer
        self.use_residual = use_residual
        self.use_softplus_decay = use_softplus_decay
        self.norm_after_flatten = norm_after_flatten

        self.num_q_heads = num_q_heads
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.num_f_heads = num_f_heads
        self.num_g_heads = num_g_heads
        self.num_weight_heads = num_weight_heads

        self.num_heads = max(num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads)

        divide_if_divisible(self.num_heads, self.num_q_heads)
        divide_if_divisible(self.num_heads, self.num_k_heads)
        divide_if_divisible(self.num_heads, self.num_v_heads)
        divide_if_divisible(self.num_heads, self.num_f_heads)
        divide_if_divisible(self.num_heads, self.num_weight_heads)
        divide_if_divisible(self.num_heads, self.num_g_heads)

        self.q_shape = self.num_q_heads * self.k_head_dim
        self.k_shape = self.num_k_heads * self.k_head_dim
        self.v_shape = self.num_v_heads * self.v_head_dim
        self.g_shape = self.num_g_heads * self.v_head_dim

        self.conv_dim = self.q_shape + self.k_shape + self.v_shape

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)

        self.input_projection = ParameterizedLinear(
            self.input_size, self.conv_dim + self.num_f_heads + self.g_shape, bias=add_bias, std=std
        )

        if self.use_softplus_decay:
            self.decay_gate = SoftplusDecayGate(0, self.num_heads, std=None, has_projection=False)

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

            mark_parameter_as_mup_learning_rate(self.conv1d.weight)

        if self.use_residual:
            self.D = nn.Parameter(torch.empty(self.num_heads, self.v_head_dim))
            mark_parameter_as_no_weight_decay(self.D)

        self.state_weight = nn.Parameter(torch.empty(self.num_weight_heads, self.v_head_dim, self.v_head_dim))

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.output_projection = ParameterizedLinear(self.g_shape, self.output_size, bias=False, std=std)

        self.g_norm = get_normalization_function(
            normalization_function, (self.num_heads if self.norm_after_flatten else 1) * self.v_head_dim
        )

        mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        mark_parameter_as_mup_learning_rate(self.state_weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)

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

        conv_state, rsa_state = (None, None) if cache_params is None else cache_params.get_cache(self.layer_idx)

        x = self.input_projection(x)
        x, f, g = x.split((self.conv_dim, self.num_f_heads, self.g_shape), dim=-1)

        if self.use_softplus_decay:
            f = self.decay_gate(f, final_exponential=True)

        if self.kernel_size is not None:
            x, conv_state = causal_convolution(
                hidden_states=x,
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

        q, k, v = x.split((self.q_shape, self.k_shape, self.v_shape), dim=-1)

        q = q.view(*q.size()[:-1], self.num_q_heads, self.k_head_dim)
        k = k.view(*k.size()[:-1], self.num_k_heads, self.k_head_dim)
        v = v.view(*v.size()[:-1], self.num_v_heads, self.v_head_dim)

        x, rsa_state = rsa(
            query=q,
            key=k,
            value=v,
            weight=self.state_weight,
            forget_input=f,
            input_state=rsa_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            do_gate_sigmoid=not self.use_softplus_decay,
        )

        if self.use_residual:
            x = x + v * self.D

        if cache_params is not None:
            cache_params.update(
                conv_state=conv_state, ssm_state=rsa_state, num_tokens_added=x.size(1), layer_idx=self.layer_idx
            )

        if self.norm_after_flatten:
            x = x.flatten(-2, -1)
            g = g.repeat_interleave(self.num_heads // self.num_g_heads, dim=-1)
            x = x * F.silu(g)
            x = self.g_norm(x)
        else:
            g = g.view(*g.size()[:-1], -1, self.v_head_dim)
            g = g.repeat_interleave(self.num_heads // self.num_g_heads, dim=-2)
            x = x * F.silu(g)
            x = self.g_norm(x)
            x = x.flatten(-2, -1)

        x = self.output_projection(x)

        if not self.use_padding_free_transformer and attention_mask is not None:
            x = unpack_sequence(inputs=x, cu_seqlens=cu_seqlens, output_shape=(B, S, *x.size()[1:]))

        return x

    @torch.no_grad()
    def reset_parameters(self) -> None:
        W = torch.eye(self.v_head_dim)
        W = W[None, ...].expand(self.num_heads, -1, -1)
        self.state_weight.copy_(W)

        if self.use_residual:
            nn.init.ones_(self.D)

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping}\nweight_shape: {str(self.state_weight.shape)}"
