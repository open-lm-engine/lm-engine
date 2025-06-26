# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .....utils import divide_if_divisible
from ....cache import GenerationCache
from ....modeling_utils import Attention, apply_rotary_pos_emb, repeat_key_value
from ....modeling_utils.mlp_blocks.mlp import _get_std_for_linear
from ..linear import DesyncResidualLinear


class DesyncResidualAttention(Attention):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        pretraining_tensor_parallel_size: int,
        attention_multiplier: float,
        attention_head_type: str,
        position_embedding_type: str,
        add_bias: bool,
        softmax_dropout: float,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        m_residual: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
        all_reduce: bool,
    ) -> None:
        nn.Module.__init__(self)

        self.tp_world_size = pretraining_tensor_parallel_size

        self.causal = causal
        self.global_hidden_size = hidden_size
        self.global_num_heads = num_attention_heads
        self.global_num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias
        self.curent_attention_all_reduce = all_reduce

        initializer_range = initializer_range
        self.m_residual = m_residual
        num_layers = num_layers

        divide_if_divisible(
            self.global_hidden_size,
            self.global_num_heads,
            f"`embed_dim` ({self.global_hidden_size}) must be divisible by `num_heads` ({self.global_num_heads})",
        )

        self.hidden_size = divide_if_divisible(
            self.global_hidden_size, self.tp_world_size, "hidden_size should be divisible by TP world size"
        )

        self.num_heads = divide_if_divisible(
            self.global_num_heads, self.tp_world_size, "num_heads must be divisible by TP world size"
        )

        self.head_dim = divide_if_divisible(self.hidden_size, self.num_heads, "")
        self.attention_head_type = attention_head_type
        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier

        self.layer_idx = layer_idx

        if self.attention_head_type == "mha":
            if self.global_num_key_value_heads is None:
                self.global_num_key_value_heads = self.global_num_heads

            assert (
                self.global_num_heads == self.global_num_key_value_heads
            ), f"{self.__class__.__name__} should have same number of heads for query, keys and values"

            self.num_key_value_heads = self.num_heads
        elif self.attention_head_type == "gqa":
            assert (
                self.global_num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            assert self.global_num_key_value_heads > 1, (
                "GroupedQueryAttention should have more than 1 head for keys and values, use MultiQueryAttention class if "
                "you want to use 1 head for keys and values"
            )

            divide_if_divisible(
                self.global_num_heads,
                self.global_num_key_value_heads,
                f"`num_heads` ({self.global_num_heads}) should be a multiple of `num_key_value_heads` ({self.global_num_key_value_heads})",
            )

            self.num_key_value_heads = divide_if_divisible(
                self.global_num_key_value_heads,
                self.tp_world_size,
                f"`num_key_value_heads` ({self.global_num_key_value_heads}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})",
            )
        elif self.attention_head_type == "mqa":
            raise ValueError("mqa is not supported with DesyncResidualAttention")
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        # note that the actual layout is different for the output and depends on whether we are using MHA, MQA or GQA
        # (self.hidden_size + 2 * self.num_key_value_heads * self.head_dim) is just the actual number output features
        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.c_attn = DesyncResidualLinear(
            self.global_hidden_size,
            self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
            tensor_parallel_size=self.tp_world_size,
            bias=self.add_bias,
            std=std,
        )

        self.c_proj = DesyncResidualLinear(
            self.hidden_size,
            self.global_hidden_size,
            tensor_parallel_size=self.tp_world_size,
            bias=self.add_bias,
            std=std / math.sqrt(2 * num_layers),
        )

        self.softmax_dropout = nn.Identity() if softmax_dropout == 0 else nn.Dropout(softmax_dropout)

        assert dropout == 0, "residual dropout is not supported with DesyncResidual"
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

    def _prepare_qkv_for_forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ==========================================================================================
        # hidden_states -> (1, batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        hidden_states = self.c_attn(hidden_states)

        # ==========================================================================================
        # hidden_states -> (TP, batch_size, query_length, [num_heads + num_key_value_heads * 2] * head_dim)
        # ==========================================================================================

        tp, batch_size, query_length, _ = hidden_states.shape
        hidden_states = hidden_states.view(tp * batch_size, query_length, -1)

        # ==========================================================================================
        # hidden_states -> (TP * batch_size, query_length, [num_heads + num_key_value_heads * 2] * head_dim)
        # ==========================================================================================

        # for MHA, we can get away with doing just 1 transpose which is not true for GQA
        if self.attention_head_type == "mha":
            query, key, value = self._prepare_qkv_for_forward_mha(hidden_states)
        elif self.attention_head_type == "gqa":
            query, key, value = self._prepare_qkv_for_forward_gqa(hidden_states)
        elif self.attention_head_type == "mqa":
            raise NotImplementedError("mqa doesn't work with DesyncResidual")
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        # ==========================================================================================
        # query -> (TP * batch_size, num_heads, query_length, head_dim)
        # key -> (TP * batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (TP * batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        # ==========================================================================================
        # hidden_states -> (1, batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        # ==========================================================================================
        # query -> (TP * batch_size, num_heads, query_length, head_dim)
        # key -> (TP * batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (TP * batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        if self.position_embedding_type == "rope":
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        # ==========================================================================================
        # query -> (TP * batch_size, num_heads, query_length, head_dim)
        # key -> (TP * batch_size, num_key_value_heads, key_length, head_dim)
        # value -> (TP * batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
        value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

        # ==========================================================================================
        # query -> (TP * batch_size, num_heads, query_length, head_dim)
        # key -> (TP * batch_size, num_heads, key_length, head_dim)
        # value -> (TP * batch_size, num_heads, key_length, head_dim)
        # ==========================================================================================

        if attention_mask is not None:
            # TODO avoid this repeat on every layer
            attention_mask = attention_mask.repeat(self.tp_world_size, 1, 1, 1)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.softmax_dropout_p if self.training else 0,
            is_causal=self.causal if attention_mask is None else False,
            scale=self._get_softmax_scale(),
            enable_gqa=True,
        )

        del query, key, value

        # ==========================================================================================
        # hidden_states -> (TP * batch_size, num_heads, query_length, head_dim)
        # ==========================================================================================

        query_length = hidden_states.shape[2]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(self.tp_world_size, -1, query_length, self.num_heads * self.head_dim)

        # ==========================================================================================
        # hidden_states -> (TP, batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        hidden_states = self.c_proj(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        if self.curent_attention_all_reduce:
            hidden_states = hidden_states + residual / self.tp_world_size
            hidden_states = hidden_states.sum(dim=0)
        else:
            hidden_states = hidden_states + residual

        hidden_states = self.dropout(hidden_states)
        return hidden_states
