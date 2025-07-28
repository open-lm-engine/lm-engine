# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ..config import GPTCrossLayerConfig
from .base import CrossLayerAttention, KeyValueProjection


def get_sequence_mixer(
    config: GPTCrossLayerConfig, causal: bool, use_padding_free_transformer: bool, layer_idx: int
) -> CrossLayerAttention:
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
        use_padding_free_transformer=use_padding_free_transformer,
    )
