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


if is_fla_available():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size,
        k_head_dim,
        v_head_dim,
        num_k_heads,
        num_v_heads,
        use_gate: bool,
        allow_neg_eigval,
        conv_size: int,
        layer_idx: int,
        norm_eps: float,
        init_method: str,
        initializer_range: float,
        num_layers: int,
        use_padding_free_transformer: bool,
    ) -> GatedDeltaNet:
        super().__init__()

        assert not use_padding_free_transformer

        self.mode = "chunk"
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size

        self.use_gate = use_gate
        self.conv_size = conv_size

        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim

        self.key_dim = self.num_k_heads * self.k_head_dim
        self.value_dim = self.num_v_heads * self.v_head_dim
        self.layer_idx = layer_idx

        divide_if_divisible(self.num_v_heads, self.num_k_heads)

        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        assert init_method == "normal"
        std = initializer_range

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

        std = initializer_range / math.sqrt(2 * num_layers)

        self.o_norm = get_normalization_function("rmsnorm", self.v_head_dim, eps=norm_eps)
        self.o_proj = ParameterizedLinear(self.value_dim, hidden_size, bias=False, std=std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = "fused_recurrent" if q_len <= 64 else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = None
        if cache_params is not None and len(cache_params) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        use_cache = cache_params is not None

        qkv = self.qkv_proj(hidden_states)

        if self.use_gate:
            a, b, gate = self.ab_proj(hidden_states).split(
                (self.num_v_heads, self.num_v_heads, self.value_dim), dim=-1
            )
        else:
            a, b = self.ab_proj(hidden_states).chunk(2, dim=-1)

        conv_state_qkv = None
        if last_state is not None:
            conv_state_qkv = last_state["conv_state"]

        qkv, conv_state_qkv = causal_convolution(
            hidden_states=qkv,
            input_state=conv_state_qkv,
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

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if cache_params is not None:
            cache_params.update(
                recurrent_state=recurrent_state,
                conv_state=conv_state_qkv,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_gate:
            g = gate.view(*gate.size()[:-1], -1, self.v_head_dim)
            o = o * F.silu(g)

        o = self.o_norm(o)
        o = o.flatten(-2, -1)
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o
