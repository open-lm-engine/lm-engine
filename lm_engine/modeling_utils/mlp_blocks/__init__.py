# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from ...model_config import CommonConfig
from .mlp import MLP, MLPArgs, interleave_up_gate_tensor_for_mlp, split_up_gate_tensor_for_mlp
from .moe import MoE, MoEArgs, ParameterizedExperts


def get_mlp_block(
    config: CommonConfig, use_padding_free_transformer: bool, sequence_parallel: bool, layer_idx: int
) -> MLP | MoE:
    block = config.mlp_blocks[layer_idx]
    mlp_type = block.mlp_type

    kwargs = dict(
        hidden_size=config.hidden_size,
        config=block,
        init_method=config.init_method,
        initializer_range=config.initializer_range,
        m_width=config.m_width,
        num_layers=config.num_layers,
        use_depth_scaled_init=config.use_depth_scaled_init,
        use_padding_free_transformer=use_padding_free_transformer,
        sequence_parallel=sequence_parallel,
    )

    if mlp_type == "MLP":
        mlp = MLP(**kwargs)
    elif mlp_type == "MoE":
        mlp = MoE(**kwargs)
    else:
        raise ValueError(f"invalid mlp_type ({mlp_type}) for layer ({layer_idx})")

    return mlp
