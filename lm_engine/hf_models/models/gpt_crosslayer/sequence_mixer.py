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
from ....utils import divide_if_divisible
from ...mask import AttentionMaskInfo
from ...modeling_utils import ParameterizedLinear, apply_rotary_pos_emb, flash_attention, get_normalization_function
from .config import GPTCrossLayerConfig


class CrossLayerAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_multiplier: float,
        position_embedding_type: str,
        add_bias: bool,
        softmax_dropout: float,
        dropout: float,
        initializer_range: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
    ) -> CrossLayerAttention:
        super().__init__()

        self.causal = causal
        self.mask_value = None
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias

        assert (
            self.hidden_size % self.num_heads == 0
        ), f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.hidden_size // self.num_heads
        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier

        self.layer_idx = layer_idx

        self.q_attn = ParameterizedLinear(
            self.hidden_size, self.hidden_size, bias=self.add_bias, std=initializer_range
        )
        self.c_proj = ParameterizedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=self.add_bias,
            std=initializer_range / math.sqrt(2 * num_layers),
        )

        self.softmax_dropout_p = softmax_dropout
        self.softmax_dropout = nn.Identity() if softmax_dropout == 0 else nn.Dropout(softmax_dropout)
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

        assert (
            self.num_key_value_heads is not None
        ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

        assert self.num_heads % self.num_key_value_heads == 0, (
            f"`num_heads` ({self.num_heads}) should be a multiple of `num_key_value_heads` "
            f"({self.num_key_value_heads})"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_info: AttentionMaskInfo,
        key: torch.Tensor,
        value: torch.Tensor,
        rope_cos_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if is_kernel_allowed(Kernel.flash_attention_2) or is_kernel_allowed(Kernel.flash_attention_3):
            query = self.q_attn(hidden_states)
            query = query.view(*hidden_states.size()[:-1], self.num_heads, -1)

            if self.use_padding_free_transformer:
                if self.position_embedding_type == "rope":
                    query = apply_rotary_pos_emb(query, rope_cos_sin)

                hidden_states = flash_attention(
                    q=query,
                    k=key,
                    v=value,
                    attention_mask_info=attention_mask_info,
                    causal=self.causal,
                    dropout=self.softmax_dropout_p if self.training else 0,
                    softmax_scale=self.attention_multiplier,
                )
            else:
                if self.position_embedding_type == "rope":
                    # TODO avoid this extra transpose
                    query = query.transpose(1, 2)
                    query = apply_rotary_pos_emb(query, rope_cos_sin)
                    query = query.transpose(1, 2)

                hidden_states = flash_attention(
                    q=query,
                    k=key,
                    v=value,
                    attention_mask_info=attention_mask_info,
                    causal=self.causal,
                    dropout=self.softmax_dropout_p if self.training else 0,
                    softmax_scale=self.attention_multiplier,
                )
        else:
            query = query.view(*query.size()[:-1], self.num_heads, -1)
            query = query.transpose(1, 2)

            if self.position_embedding_type == "rope":
                query = apply_rotary_pos_emb(query, rope_cos_sin)

            attention_mask = attention_mask_info.get_attention_mask()

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

            hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states.flatten(-2, -1)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class KeyValueProjection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        add_bias: bool,
        initializer_range: float,
        normalization_function: str,
        layer_norm_epsilon: float,
    ) -> KeyValueProjection:
        super().__init__()

        self.num_key_value_heads = num_key_value_heads
        head_dim = divide_if_divisible(hidden_size, num_attention_heads, "")

        self.ln = get_normalization_function(normalization_function, hidden_size, layer_norm_epsilon)
        self.kv_attn = ParameterizedLinear(
            hidden_size,
            2 * self.num_key_value_heads * head_dim,
            bias=add_bias,
            std=initializer_range,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.ln(hidden_states)
        hidden_states = self.kv_attn(hidden_states)

        hidden_states = hidden_states.view(*hidden_states.size()[:-1], self.num_key_value_heads, -1)
        key, value = hidden_states.chunk(2, -1)

        return key, value


def get_sequence_mixer(config: GPTCrossLayerConfig, causal: bool, layer_idx: int) -> CrossLayerAttention:
    block = config.sequence_mixer_blocks[layer_idx]
    assert block.sequence_mixer_type == "softmax_attention"

    return CrossLayerAttention(
        hidden_size=config.hidden_size,
        num_attention_heads=block.num_attention_heads,
        num_key_value_heads=block.num_key_value_heads,
        attention_multiplier=block.attention_multiplier,
        position_embedding_type=config.position_embedding_type,
        add_bias=block.add_bias,
        softmax_dropout=block.softmax_dropout,
        dropout=block.dropout,
        initializer_range=config.initializer_range,
        num_layers=config.num_layers,
        causal=causal,
        layer_idx=layer_idx,
    )
