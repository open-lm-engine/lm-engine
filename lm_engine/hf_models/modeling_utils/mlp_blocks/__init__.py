# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from ...config import CommonConfig
from .delta_mlp import DeltaMLP
from .mlp import MLP, interleave_up_gate_tensor_for_mlp, split_up_gate_tensor_for_mlp
from .moe import MoE, ParameterizedExperts


def get_mlp_block(
    config: CommonConfig, use_padding_free_transformer: bool, sequence_parallel: bool, layer_idx: int
) -> MLP | MoE | DeltaMLP:
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
        use_depth_scaled_init=config.use_depth_scaled_init,
        use_padding_free_transformer=use_padding_free_transformer,
        sequence_parallel=sequence_parallel,
    )

    if mlp_type == "MLP":
        mlp = MLP(**kwargs)
    elif mlp_type == "MoE":
        mlp = MoE(
            **kwargs,
            shared_intermediate_size=block.shared_intermediate_size,
            shared_expert_gating=block.shared_expert_gating,
            normalized_topk=block.normalized_topk,
            num_experts=block.num_experts,
            num_experts_per_tok=block.num_experts_per_tok,
        )
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
            use_head_o_norm=block.use_head_o_norm,
            allow_neg_eigval=block.allow_neg_eigval,
            conv_size=block.kernel_size,
            layer_idx=layer_idx,
            norm_eps=config.layer_norm_epsilon,
            A_init_min=block.A_init_min,
            A_init_max=block.A_init_max,
            dt_init_min=block.dt_init_min,
            dt_init_max=block.dt_init_max,
            dt_init_floor=block.dt_init_floor,
            value_scale=block.value_scale,
            use_v_silu=block.use_v_silu,
            use_v_norm=block.use_v_norm,
            use_b_proj_per_row_hyperball=block.use_b_proj_per_row_hyperball,
        )
    else:
        raise ValueError(f"invalid mlp_type ({mlp_type}) for layer ({layer_idx})")

    return mlp
