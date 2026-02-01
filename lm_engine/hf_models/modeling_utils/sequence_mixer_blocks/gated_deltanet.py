# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....utils import divide_if_divisible, is_fla_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_no_weight_decay
from ..convolution import ParameterizedConv1d
from ..linear import ParameterizedLinear
from ..normalization import get_normalization_function
from .causal_convolution import causal_convolution
from .utils import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence


if is_fla_available():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        k_head_dim: int,
        v_head_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        use_gate: bool,
        attention_multiplier: float | None,
        allow_neg_eigval: bool,
        conv_size: int,
        layer_idx: int,
        norm_eps: float,
        init_method: str,
        initializer_range: float,
        m_width: float | None,
        num_layers: int,
        use_padding_free_transformer: bool,
    ) -> GatedDeltaNet:
        super().__init__()

        assert not use_padding_free_transformer
        self.use_padding_free_transformer = use_padding_free_transformer

        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size

        self.use_gate = use_gate
        self.conv_size = conv_size

        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads

        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim

        self.key_dim = self.num_k_heads * self.k_head_dim
        self.value_dim = self.num_v_heads * self.v_head_dim
        self.layer_idx = layer_idx

        self.attention_multiplier = attention_multiplier

        divide_if_divisible(self.num_v_heads, self.num_k_heads)

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.qkv_proj = ParameterizedLinear(hidden_size, 2 * self.key_dim + self.value_dim, bias=False, std=std)

        self.ab_proj = ParameterizedLinear(
            hidden_size, 2 * self.num_v_heads + (self.value_dim if use_gate else 0), bias=False, std=std
        )

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        self.dt_bias = nn.Parameter(inv_dt)
        mark_parameter_as_no_weight_decay(self.dt_bias)

        self.conv_size = conv_size
        self.qkv_conv1d = ParameterizedConv1d(
            in_channels=2 * self.key_dim + self.value_dim,
            out_channels=2 * self.key_dim + self.value_dim,
            kernel_size=conv_size,
            padding=conv_size - 1,
            groups=2 * self.key_dim + self.value_dim,
            bias=False,
            std=std,  # TODO
        )
        self.activation_string = "silu"

        self.o_norm = get_normalization_function("rmsnorm", self.v_head_dim, eps=norm_eps)

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.o_proj = ParameterizedLinear(self.value_dim, hidden_size, bias=False, std=std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        c, h = (None, None) if cache_params is None else cache_params.get_cache(self.layer_idx)
        use_cache = cache_params is not None

        qkv = self.qkv_proj(hidden_states)

        if self.use_gate:
            a, b, gate = self.ab_proj(hidden_states).split(
                (self.num_v_heads, self.num_v_heads, self.value_dim), dim=-1
            )
        else:
            a, b = self.ab_proj(hidden_states).chunk(2, dim=-1)

        qkv, c = causal_convolution(
            hidden_states=qkv,
            input_state=c,
            attention_mask=attention_mask,
            conv1d_weight=self.qkv_conv1d.weight,
            conv1d_bias=self.qkv_conv1d.bias,
            conv1d_num_groups=qkv.size(-1),
            return_cache_state=cache_params is not None,
            activation_string=self.activation_string,
            conv1d_padding=self.conv_size - 1,
            conv1d_stride=1,
        )

        q, k, v = qkv.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)

        q_size = q.size()
        q = q.view(*q_size[:-1], -1, self.k_head_dim)
        k = k.view(*q_size[:-1], -1, self.k_head_dim)
        v = v.view(*v.size()[:-1], -1, self.v_head_dim)

        if self.num_v_heads > self.num_k_heads:
            q = q.repeat_interleave(repeats=self.num_v_heads // self.num_k_heads, dim=-2)
            k = k.repeat_interleave(repeats=self.num_v_heads // self.num_k_heads, dim=-2)

        beta = b.sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.use_padding_free_transformer:
            assert cache_params is None
            assert attention_mask is None
        else:
            assert cu_seqlens is None
            assert max_seqlen is None

            B, S = q.size()[:2]

            if attention_mask is not None:
                cu_seqlens, max_seqlen = compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask)
                q, k, v, g, beta = pack_sequence(inputs=(q, k, v, g, beta), cu_seqlens=cu_seqlens)

        # change to inference mode.
        mode = "fused_recurrent" if S <= 64 else "chunk"
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        if mode == "chunk":
            o, h = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=self.attention_multiplier,
                initial_state=h,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == "fused_recurrent":
            o, h = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=self.attention_multiplier,
                initial_state=h,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if not self.use_padding_free_transformer and attention_mask is not None:
            o = unpack_sequence(inputs=o, cu_seqlens=cu_seqlens, output_shape=(B, S, *hidden_states.size()[1:]))

        if cache_params is not None:
            cache_params.update(
                conv_state=c, ssm_state=h, num_tokens_added=hidden_states.size(1), layer_idx=self.layer_idx
            )

        if self.use_gate:
            g = gate.view(*gate.size()[:-1], -1, self.v_head_dim)
            o = o * F.silu(g)

        o = self.o_norm(o)
        o = o.flatten(-2, -1)
        o = self.o_proj(o)

        return o
