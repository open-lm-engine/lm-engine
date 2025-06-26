# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math

import torch
import torch.nn as nn

from ....utils import ProcessGroupManager, divide_if_divisible
from ...modeling_utils import Attention, get_attention_head_type
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
    ) -> None:
        nn.Module.__init__(self)

        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        self.causal = causal
        self.global_hidden_size = hidden_size
        self.global_num_heads = num_attention_heads
        self.global_num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias
        self.use_padding_free_transformer = use_padding_free_transformer

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
        self.attention_head_type = get_attention_head_type(self.global_num_heads, self.global_num_key_value_heads)
        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        if self.attention_head_type == "mha":
            if self.global_num_key_value_heads is None:
                self.global_num_key_value_heads = self.global_num_heads

            assert (
                self.global_num_heads == self.global_num_key_value_heads
            ), f"{self.__class__.__name__} should have same number of heads for query, keys and values"

            self.num_key_value_heads = self.num_heads

            self.c_attn = ColumnParallelLinear(
                self.global_hidden_size,
                self.global_hidden_size + 2 * self.global_num_key_value_heads * self.head_dim,
                bias=self.add_bias,
                std=std,
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )
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
        elif self.attention_head_type == "mqa":
            if self.global_num_key_value_heads is None:
                self.global_num_key_value_heads = 1

            assert (
                self.global_num_key_value_heads == 1
            ), f"{self.__class__.__name__} should have 1 head for keys and values"

            self.num_key_value_heads = 1

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
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

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

    def _prepare_qkv_for_forward_mqa(
        self, query_key_value: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = query_key_value

        if self.use_padding_free_transformer:
            total_q = query.shape[0]

            query = query.view(total_q, self.num_heads, -1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
        else:
            batch_size, query_length = query.shape[:-1]
            query = query.view(batch_size, query_length, self.num_heads, -1)

            query = query.transpose(1, 2)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        return query, key, value


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
