# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ..config import DesyncResidualConfig
from .base import DesyncResidualAttention


def get_sequence_mixer(
    config: DesyncResidualConfig, causal: bool, use_padding_free_transformer: bool, layer_idx: int
) -> DesyncResidualAttention:
    if use_padding_free_transformer:
        raise NotImplementedError("padding free transformer is not implemented with DesyncResidual")

    block = config.sequence_mixer_blocks[layer_idx]

    return DesyncResidualAttention(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=block.num_key_value_heads,
        pretraining_tensor_parallel_size=config.pretraining_tensor_parallel_size,
        attention_multiplier=block.attention_multiplier,
        attention_head_type=block.attention_head_type,
        position_embedding_type=config.position_embedding_type,
        add_bias=block.add_bias,
        softmax_dropout=block.softmax_dropout,
        dropout=block.dropout,
        init_method=config.init_method,
        initializer_range=config.initializer_range,
        m_width=config.m_width,
        m_residual=config.m_residual,
        num_layers=config.num_layers,
        causal=causal,
        layer_idx=layer_idx,
        all_reduce=config.reduce_pattern[layer_idx]["attention"],
    )
