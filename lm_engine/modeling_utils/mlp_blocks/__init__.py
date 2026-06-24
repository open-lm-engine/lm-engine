# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import TYPE_CHECKING

from .delta_mlp import DeltaMLP, DeltaMLPArgs
from .mlp import MLP, MLPArgs, interleave_up_gate_tensor_for_mlp, split_up_gate_tensor_for_mlp
from .moe import MoE, MoEArgs, ParameterizedExperts


if TYPE_CHECKING:
    from ...model_config import CommonConfig


def get_mlp_block(
    config: CommonConfig, use_padding_free_transformer: bool, sequence_parallel: bool, layer_idx: int
) -> MLP | MoE | DeltaMLP:
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
    elif mlp_type == "DeltaMLP":
        mlp = DeltaMLP(
            **kwargs,
            num_ranks=block.num_ranks,
            num_heads=block.num_heads,
            use_v_proj=block.use_v_proj,
            use_q_l2norm=block.use_q_l2norm,
            use_shortconv=block.use_shortconv,
            use_tied_beta=block.use_tied_beta,
            use_decay_beta=block.use_decay_beta,
            allow_neg_eigval=block.allow_neg_eigval,
            use_o_norm=block.use_o_norm,
            conv_size=block.kernel_size,
            layer_idx=layer_idx,
            norm_eps=config.layer_norm_epsilon,
            A_init_min=block.A_init_min,
            A_init_max=block.A_init_max,
            dt_init_min=block.dt_init_min,
            dt_init_max=block.dt_init_max,
            dt_init_floor=block.dt_init_floor,
            value_scale=block.value_scale,
            use_b_proj_per_row_hyperball=block.use_b_proj_per_row_hyperball,
        )
    else:
        raise ValueError(f"invalid mlp_type ({mlp_type}) for layer ({layer_idx})")

    return mlp
