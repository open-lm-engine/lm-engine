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
from ....utils import ProcessGroupManager, divide_if_divisible, print_ranks_all
from ...cache import GenerationCache
from ...modeling_utils import Attention, apply_rotary_pos_emb, flash_attention
from ...modeling_utils.mlp_blocks.mlp import _get_std_for_linear
from ..dropout import Dropout_TP
from ..linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear


class Attention_TP(Attention):
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
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        causal: bool,
        layer_idx: int | None = None,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> Attention_TP:
        nn.Module.__init__(self)

        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        self.causal = causal
        self.global_hidden_size = hidden_size
        self.global_num_heads = num_attention_heads
        self.global_num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias
        self.use_padding_free_transformer = use_padding_free_transformer
        self.sequence_parallel = sequence_parallel

        divide_if_divisible(
            self.global_hidden_size,
            self.global_num_heads,
            f"`embed_dim` ({self.global_hidden_size}) must be divisible by `num_heads` ({self.global_num_heads})",
        )

        self.hidden_size = divide_if_divisible(
            self.global_hidden_size, tp_world_size, "hidden_size should be divisible by TP world size"
        )

        self.num_heads = divide_if_divisible(
            self.global_num_heads, tp_world_size, "num_heads must be divisible by TP world size"
        )

        self.head_dim = divide_if_divisible(self.hidden_size, self.num_heads, "")
        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        if self.global_num_heads > 1 and self.global_num_key_value_heads == 1:
            # MQA
            self.num_key_value_heads = 1
            self.is_mqa = True

            self.c_attn = _MQA_QueryKeyValueProjection(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                add_bias=add_bias,
                m_width=m_width,
                num_layers=num_layers,
                init_method=init_method,
                initializer_range=initializer_range,
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )
        else:
            self.is_mqa = False

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
                tp_world_size,
                f"`num_key_value_heads` ({self.global_num_key_value_heads}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
            )

            self.c_attn = ColumnParallelLinear(
                self.global_hidden_size,
                self.global_hidden_size + 2 * self.global_num_key_value_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )

        self.c_proj = RowParallelLinear(
            self.global_hidden_size,
            self.global_hidden_size,
            bias=self.add_bias,
            std=std / math.sqrt(2 * num_layers),
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        self.softmax_dropout_p = softmax_dropout

        self.softmax_dropout = (
            nn.Identity()
            if softmax_dropout == 0
            else Dropout_TP(
                softmax_dropout,
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )
        )
        self.dropout = (
            nn.Identity()
            if dropout == 0
            else Dropout_TP(
                dropout,
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        use_flash_attention_2 = is_kernel_allowed(Kernel.flash_attention_2)
        use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)

        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        if self.use_padding_free_transformer:
            assert use_flash_attention_2 or use_flash_attention_3
            assert past_key_values is None

            total_q = hidden_states.shape[0] * (tp_world_size if self.sequence_parallel else 1)
            input_shape = (total_q, self.num_key_value_heads, -1)
            output_shape = (total_q, -1, self.head_dim)
        else:
            batch_size, query_length = hidden_states.shape[:-1]
            query_length *= tp_world_size if self.sequence_parallel else 1

            input_shape = (batch_size, query_length, self.num_key_value_heads, -1)
            output_shape = (batch_size, query_length, -1, self.head_dim)

        hidden_states = self.c_attn(hidden_states)

        if self.is_mqa:
            query, key, value = hidden_states

            if self.use_padding_free_transformer:
                query = query.view(total_q, -1, self.head_dim)
            else:
                query = query.view(batch_size, query_length, -1, self.head_dim)

            key = key.unsqueeze(-2)
            value = value.unsqueeze(-2)
        else:
            hidden_states = hidden_states.view(*input_shape)

            query, key, value = hidden_states.split(
                ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim), dim=-1
            )

            query = query.reshape(*output_shape)

        if not self.use_padding_free_transformer:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        if self.position_embedding_type == "rope":
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key_states=key, value_states=value, layer_idx=self.layer_idx)

        if use_flash_attention_2 or use_flash_attention_3:
            if self.use_padding_free_transformer:
                output_shape = (-1, self.hidden_size)
            else:
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

                output_shape = (batch_size, query_length, -1)

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
                use_padding_free_transformer=self.use_padding_free_transformer,
                causal=self.causal,
                dropout=self.softmax_dropout_p if self.training else 0,
                softmax_scale=self.attention_multiplier,
            )

            del query, key, value

            hidden_states = wait_for_ACT(hidden_states, wait_in_forward=False, wait_in_backward=True)
            hidden_states = hidden_states.view(*output_shape)
        else:
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


class _MQA_QueryKeyValueProjection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        add_bias: bool,
        m_width: int,
        num_layers: int,
        init_method: str,
        initializer_range: float,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.global_hidden_size = hidden_size
        self.add_bias = add_bias

        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        hidden_size = divide_if_divisible(
            self.global_hidden_size, tp_world_size, "hidden_size should be divisible by TP world size"
        )

        num_heads = divide_if_divisible(
            num_attention_heads, tp_world_size, "num_heads must be divisible by TP world size"
        )
        self.head_dim = divide_if_divisible(hidden_size, num_heads, "")

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.q_attn = ColumnParallelLinear(
            self.global_hidden_size,
            self.global_hidden_size,
            bias=self.add_bias,
            std=std,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        self.kv_attn = ReplicatedLinear(
            self.global_hidden_size,
            2 * self.head_dim,
            bias=self.add_bias,
            std=std / math.sqrt(2 * num_layers),
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.q_attn(hidden_states)

        key_value = self.kv_attn(hidden_states)
        key, value = key_value.chunk(2, -1)

        return query, key, value
