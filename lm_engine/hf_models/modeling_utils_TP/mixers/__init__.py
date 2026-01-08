# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...config import CommonConfig
from .attention import Attention_TP
from .mlp import MLP_TP
from .moe import MoE_TP


def get_sequence_mixer_TP(
    config: CommonConfig,
    causal: bool,
    use_padding_free_transformer: bool,
    layer_idx: int,
    sequence_parallel: bool,
) -> Attention_TP:
    block = config.sequence_mixer_blocks[layer_idx]
    sequence_mixer_type = block.sequence_mixer_type

    sequence_mixer_kwargs = dict(
        hidden_size=config.hidden_size,
        num_attention_heads=block.num_attention_heads,
        num_key_value_heads=block.num_key_value_heads,
        attention_multiplier=block.attention_multiplier,
        position_embedding_type=config.position_embedding_type,
        add_bias=block.add_bias,
        softmax_dropout=block.softmax_dropout,
        dropout=block.dropout,
        init_method=config.init_method,
        initializer_range=config.initializer_range,
        m_width=config.m_width,
        num_layers=config.num_layers,
        causal=causal,
        layer_idx=layer_idx,
        sequence_parallel=sequence_parallel,
    )

    if sequence_mixer_type == "softmax_attention":
        return Attention_TP(**sequence_mixer_kwargs, use_padding_free_transformer=use_padding_free_transformer)


def get_mlp_block_TP(
    config: CommonConfig, use_padding_free_transformer: bool, sequence_parallel: bool, layer_idx: int
) -> MLP_TP | MoE_TP:
    block = config.mlp_blocks[layer_idx]
    mlp_type = block.mlp_type

    kwargs = dict(
        hidden_size=config.hidden_size,
        intermediate_size=block.intermediate_size,
        activation_function=block.activation_function,
        add_bias=block.add_bias,
        dropout=block.dropout,
        init_method=config.init_method,
        initializer_range=config.initializer_range,
        m_width=config.m_width,
        num_layers=config.num_layers,
        use_padding_free_transformer=use_padding_free_transformer,
        sequence_parallel=sequence_parallel,
    )

    if mlp_type == "MLP":
        mlp = MLP_TP(**kwargs)
    elif mlp_type == "MoE":
        mlp = MoE_TP(
            **kwargs,
            shared_intermediate_size=block.shared_intermediate_size,
            num_experts=block.num_experts,
            num_experts_per_tok=block.num_experts_per_tok,
        )
    else:
        raise ValueError(f"invalid mlp_type ({mlp_type}) for layer ({layer_idx})")

    return mlp
