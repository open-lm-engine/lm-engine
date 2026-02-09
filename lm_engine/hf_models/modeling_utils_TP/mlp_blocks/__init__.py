# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...config import CommonConfig
from ...modeling_utils import MLP
from .moe import MoE_TP


def get_mlp_block_TP(
    config: CommonConfig, use_padding_free_transformer: bool, sequence_parallel: bool, layer_idx: int
) -> MLP | MoE_TP:
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
        mlp = MLP(**kwargs)
    elif mlp_type == "MoE":
        mlp = MoE_TP(
            **kwargs,
            shared_intermediate_size=block.shared_intermediate_size,
            use_interleaved_weights=block.use_interleaved_weights,
            shared_expert_gating=block.shared_expert_gating,
            normalized_topk=block.normalized_topk,
            num_experts=block.num_experts,
            num_experts_per_tok=block.num_experts_per_tok,
        )
    else:
        raise ValueError(f"invalid mlp_type ({mlp_type}) for layer ({layer_idx})")

    return mlp
