# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed, wait_for_ACT
from ...cache import GenerationCache
from ..linear import ParameterizedLinear
from ..normalization import get_normalization_function
from .flash_attention_utils import flash_attention


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        query_compression_size: int,
        key_value_compression_size: int,
        num_attention_heads: int,
        head_dim: int,
        attention_multiplier: float,
        position_embedding_type: str,
        add_bias: bool,
        softmax_dropout: float,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
        normalization_function: str,
        layer_norm_epsilon: float = 1e-5,
    ) -> MultiHeadLatentAttention:
        super().__init__()

        self.causal = causal
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = head_dim
        self.add_bias = add_bias
        self.query_compression_size = query_compression_size
        self.key_value_compression_size = key_value_compression_size
        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)

        if self.position_embedding_type == "rope":
            raise NotImplementedError()
        else:
            self.query_down_projection = ParameterizedLinear(
                self.hidden_size, self.query_compression_size, bias=self.add_bias, std=std
            )

            self.query_ln = get_normalization_function(
                normalization_function, self.query_compression_size, eps=layer_norm_epsilon
            )

            self.query_up_projection = ParameterizedLinear(
                self.query_compression_size, self.num_heads * self.head_dim, bias=self.add_bias, std=std
            )

            self.key_value_down_projection = ParameterizedLinear(
                self.hidden_size,
                2 * self.key_value_compression_size,
                bias=self.add_bias,
                std=std,
            )

            self.key_value_ln = get_normalization_function(
                normalization_function, 2 * self.key_value_compression_size, eps=layer_norm_epsilon
            )

            self.key_up_projection = ParameterizedLinear(
                self.key_value_compression_size, self.num_heads * self.head_dim, bias=self.add_bias, std=std
            )

            self.value_up_projection = ParameterizedLinear(
                self.key_value_compression_size, self.num_heads * self.head_dim, bias=self.add_bias, std=std
            )

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.c_proj = ParameterizedLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=self.add_bias, std=std
        )

        self.softmax_dropout_p = softmax_dropout

        self.softmax_dropout = nn.Identity() if softmax_dropout == 0 else nn.Dropout(softmax_dropout)
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        use_flash_attention_2 = is_kernel_allowed(Kernel.flash_attention_2)
        use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)

        assert use_flash_attention_2 or use_flash_attention_3
        assert cache_params is None

        query = self.query_down_projection(hidden_states)
        query = self.query_ln(query)

        key_value = self.key_value_down_projection(hidden_states)
        key_value = self.key_value_ln(key_value)
        key, value = key_value.chunk(2, dim=-1)

        del hidden_states, key_value

        if self.position_embedding_type == "rope":
            raise NotImplementedError()
        else:
            if cache_params is not None:
                key, value = cache_params.update(
                    key_states=key.unsqueeze(1), value_states=value.unsqueeze(1), layer_idx=self.layer_idx
                )
                key = key.squeeze(1)
                value = value.squeeze(1)

            query = self.query_up_projection(query)
            key = self.key_up_projection(key)
            value = self.value_up_projection(value)

        if use_flash_attention_2 or use_flash_attention_3:
            T = query.size(0)

            query = query.view(T, self.num_heads, -1)
            key = key.view(T, self.num_heads, -1)
            value = value.view(T, self.num_heads, -1)

            query = wait_for_ACT(query, wait_in_forward=True, wait_in_backward=False)
            key = wait_for_ACT(key, wait_in_forward=True, wait_in_backward=False)
            value = wait_for_ACT(value, wait_in_forward=True, wait_in_backward=False)

            hidden_states = flash_attention(
                query=query,
                key=key,
                value=value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                attention_mask=attention_mask,
                causal=self.causal,
                dropout=self.softmax_dropout_p if self.training else 0,
                softmax_scale=self.attention_multiplier,
            )

            del query, key, value

            hidden_states = wait_for_ACT(hidden_states, wait_in_forward=False, wait_in_backward=True)
            hidden_states = hidden_states.view(-1, self.hidden_size)
        else:
            batch_size, query_length = query.size()[:-1]
            key_length = key.size(1)

            query = query.view(batch_size, query_length, self.num_heads, -1).transpose(1, 2)
            key = key.view(batch_size, key_length, self.num_heads, -1).transpose(1, 2)
            value = value.view(batch_size, key_length, self.num_heads, -1).transpose(1, 2)

            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.softmax_dropout_p if self.training else 0,
                is_causal=self.causal if attention_mask is None else False,
                scale=self.attention_multiplier,
                enable_gqa=True,
            )

            del query, key, value

            batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
