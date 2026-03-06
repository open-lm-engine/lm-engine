# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

from ...utils import SafeTensorsWeightsManager, divide_if_divisible
from ..modeling_utils import (
    interleave_query_key_value_tensor_for_attention,
    interleave_up_gate_tensor_for_mlp,
    split_query_key_value_tensor_for_attention,
    split_up_gate_tensor_for_mlp,
)
from ..models import GPTBaseConfig


def _import_granitemoehybrid_config(original_config: GraniteMoeHybridConfig, **kwargs) -> GPTBaseConfig:
    assert original_config.hidden_act == "silu"
    assert not original_config.attention_bias
    use_interleaved_weights = kwargs.pop("use_interleaved_weights", False)
    use_interleaved_weights_for_shared_experts = kwargs.pop("use_interleaved_weights_for_shared_experts", False)

    sequence_mixer_blocks = []
    for layer_idx in range(original_config.num_hidden_layers):
        layer_type = original_config.layer_types[layer_idx]

        if layer_type == "attention":
            sequence_mixer_block = {
                "sequence_mixer_type": "softmax_attention",
                "num_attention_heads": original_config.num_attention_heads,
                "num_key_value_heads": original_config.num_key_value_heads,
                "attention_multiplier": original_config.attention_multiplier,
                "add_bias": original_config.attention_bias,
                "softmax_dropout": original_config.attention_dropout,
            }
        elif layer_type == "mamba":
            sequence_mixer_block = {
                "sequence_mixer_type": "mamba2",
                "state_size": original_config.mamba_d_state,
                "intermediate_size": original_config.mamba_expand * original_config.hidden_size,
                "num_heads": original_config.mamba_n_heads,
                "conv_kernel_size": original_config.mamba_d_conv,
                "add_bias": original_config.mamba_proj_bias,
                "use_conv_bias": original_config.mamba_conv_bias,
                "num_groups": original_config.mamba_n_groups,
                "chunk_size": original_config.mamba_chunk_size,
            }
        else:
            raise ValueError(f"unexpected layer_type ({layer_type})")

        sequence_mixer_blocks.append(sequence_mixer_block)

    # Allow for 0 experts by setting mlp_blocks accordingly
    mlp_blocks = []
    for _ in range(original_config.num_hidden_layers):
        if original_config.num_local_experts == 0:
            mlp_block = {
                "mlp_type": "MLP",
                "intermediate_size": original_config.shared_intermediate_size,
                "activation_function": "swiglu",
                "add_bias": False,
                "use_interleaved_weights": use_interleaved_weights,
            }
        else:
            mlp_block = {
                "mlp_type": "MoE",
                "intermediate_size": original_config.intermediate_size,
                "shared_intermediate_size": (
                    None if original_config.shared_intermediate_size == 0 else original_config.shared_intermediate_size
                ),
                "num_experts": original_config.num_local_experts,
                "num_experts_per_tok": original_config.num_experts_per_tok,
                "activation_function": "swiglu",
                "add_bias": False,
                "use_interleaved_weights": use_interleaved_weights,
                "use_interleaved_weights_for_shared_experts": use_interleaved_weights_for_shared_experts,
            }

        mlp_blocks.append(mlp_block)

    config = GPTBaseConfig(
        vocab_size=original_config.vocab_size,
        max_position_embeddings=original_config.max_position_embeddings,
        hidden_size=original_config.hidden_size,
        num_layers=original_config.num_hidden_layers,
        position_embedding_type="nope",
        normalization_function="rmsnorm",
        layer_norm_epsilon=original_config.rms_norm_eps,
        use_cache=original_config.use_cache,
        tie_word_embeddings=original_config.tie_word_embeddings,
        initializer_range=original_config.initializer_range,
        rope_scaling=original_config.rope_scaling,
        router_aux_loss_coef=original_config.router_aux_loss_coef,
        bos_token_id=original_config.bos_token_id,
        eos_token_id=original_config.eos_token_id,
        pad_token_id=original_config.pad_token_id,
        m_emb=None if original_config.embedding_multiplier == 1 else original_config.embedding_multiplier,
        m_residual=None if original_config.residual_multiplier == 1 else original_config.residual_multiplier,
        m_width=None if original_config.logits_scaling == 1 else original_config.logits_scaling,
        sequence_mixer_blocks=sequence_mixer_blocks,
        mlp_blocks=mlp_blocks,
    )

    if use_interleaved_weights is not None:
        for block in config.mlp_blocks:
            block.use_interleaved_weights = use_interleaved_weights

    assert len(kwargs) == 0

    return config


def _import_granitemoehybrid_state_dict(
    config: GPTBaseConfig, safetensors_weights_manager: SafeTensorsWeightsManager
) -> dict:
    num_attention_heads = config.check_equal_for_all_and_get_value(
        "sequence_mixer_blocks", "num_attention_heads", sequence_mixer_type="softmax_attention"
    )

    num_key_value_heads = config.check_equal_for_all_and_get_value(
        "sequence_mixer_blocks", "num_key_value_heads", sequence_mixer_type="softmax_attention"
    )

    head_dim = divide_if_divisible(config.hidden_size, num_attention_heads, "")
    sequence_mixer_block_types = _get_sequence_mixer_block_types(config)

    state_dict = {
        "transformer.wte.weight": safetensors_weights_manager.get_tensor("model.embed_tokens.weight"),
        "transformer.ln_f.weight": safetensors_weights_manager.get_tensor("model.norm.weight"),
    }

    if safetensors_weights_manager.has_tensor("lm_head.weight"):
        state_dict["lm_head.weight"] = safetensors_weights_manager.get_tensor("lm_head.weight")

    for layer_idx in range(config.num_layers):
        use_interleaved_weights = config.mlp_blocks[layer_idx].use_interleaved_weights
        import_prefix = f"transformer.h.{layer_idx}."
        export_prefix = f"model.layers.{layer_idx}."

        state_dict[f"{import_prefix}ln_1.weight"] = safetensors_weights_manager.get_tensor(
            f"{export_prefix}input_layernorm.weight"
        )
        state_dict[f"{import_prefix}ln_2.weight"] = safetensors_weights_manager.get_tensor(
            f"{export_prefix}post_attention_layernorm.weight"
        )

        if safetensors_weights_manager.has_tensor(f"{export_prefix}block_sparse_moe.router.layer.weight"):
            state_dict[f"{import_prefix}mlp_block.gate.weight"] = safetensors_weights_manager.get_tensor(
                f"{export_prefix}block_sparse_moe.router.layer.weight"
            )

            g, u = safetensors_weights_manager.get_tensor(
                f"{export_prefix}block_sparse_moe.input_linear.weight"
            ).chunk(2, dim=1)
            state_dict[f"{import_prefix}mlp_block.c_fc.weight"] = interleave_up_gate_tensor_for_mlp(
                up_weight=u, gate_weight=g, is_interleaved=use_interleaved_weights, dim=1
            )

            state_dict[f"{import_prefix}mlp_block.c_proj.weight"] = safetensors_weights_manager.get_tensor(
                f"{export_prefix}block_sparse_moe.output_linear.weight"
            )

            if safetensors_weights_manager.has_tensor(f"{export_prefix}shared_mlp.input_linear.weight"):
                g, u = safetensors_weights_manager.get_tensor(f"{export_prefix}shared_mlp.input_linear.weight").chunk(
                    2
                )
                state_dict[f"{import_prefix}mlp_block.c_fc_shared.weight"] = interleave_up_gate_tensor_for_mlp(
                    up_weight=u,
                    gate_weight=g,
                    is_interleaved=config.mlp_blocks[layer_idx].use_interleaved_weights_for_shared_experts,
                )
                state_dict[f"{import_prefix}mlp_block.c_proj_shared.weight"] = safetensors_weights_manager.get_tensor(
                    f"{export_prefix}shared_mlp.output_linear.weight"
                )
        else:
            g, u = safetensors_weights_manager.get_tensor(f"{export_prefix}shared_mlp.input_linear.weight").chunk(2)
            state_dict[f"{import_prefix}mlp_block.c_fc.weight"] = interleave_up_gate_tensor_for_mlp(
                up_weight=u, gate_weight=g, is_interleaved=use_interleaved_weights
            )
            state_dict[f"{import_prefix}mlp_block.c_proj.weight"] = safetensors_weights_manager.get_tensor(
                f"{export_prefix}shared_mlp.output_linear.weight"
            )

        if sequence_mixer_block_types[layer_idx] == "mamba":
            state_dict[f"{import_prefix}sequence_mixer.conv1d.weight"] = safetensors_weights_manager.get_tensor(
                f"{export_prefix}mamba.conv1d.weight"
            )
            if safetensors_weights_manager.has_tensor(f"{export_prefix}mamba.conv1d.bias"):
                state_dict[f"{import_prefix}sequence_mixer.conv1d.bias"] = safetensors_weights_manager.get_tensor(
                    f"{export_prefix}mamba.conv1d.bias"
                )
            state_dict[f"{import_prefix}sequence_mixer.in_proj.weight"] = safetensors_weights_manager.get_tensor(
                f"{export_prefix}mamba.in_proj.weight"
            )
            if safetensors_weights_manager.has_tensor(f"{export_prefix}mamba.in_proj.bias"):
                state_dict[f"{import_prefix}sequence_mixer.in_proj.bias"] = safetensors_weights_manager.get_tensor(
                    f"{export_prefix}mamba.in_proj.bias"
                )
            state_dict[f"{import_prefix}sequence_mixer.decay_gate.dt_bias"] = safetensors_weights_manager.get_tensor(
                f"{export_prefix}mamba.dt_bias"
            )
            state_dict[f"{import_prefix}sequence_mixer.decay_gate.A_log"] = safetensors_weights_manager.get_tensor(
                f"{export_prefix}mamba.A_log"
            )
            state_dict[f"{import_prefix}sequence_mixer.D"] = safetensors_weights_manager.get_tensor(
                f"{export_prefix}mamba.D"
            )
            state_dict[f"{import_prefix}sequence_mixer.out_proj.weight"] = safetensors_weights_manager.get_tensor(
                f"{export_prefix}mamba.out_proj.weight"
            )
            if safetensors_weights_manager.has_tensor(f"{export_prefix}mamba.out_proj.bias"):
                state_dict[f"{import_prefix}sequence_mixer.out_proj.bias"] = safetensors_weights_manager.get_tensor(
                    f"{export_prefix}mamba.out_proj.bias"
                )
            state_dict[f"{import_prefix}sequence_mixer.norm.weight"] = safetensors_weights_manager.get_tensor(
                f"{export_prefix}mamba.norm.weight"
            )
        elif sequence_mixer_block_types[layer_idx] == "attention":
            state_dict[f"{import_prefix}sequence_mixer.c_attn.weight"] = (
                interleave_query_key_value_tensor_for_attention(
                    safetensors_weights_manager.get_slice(f"{export_prefix}self_attn.q_proj.weight"),
                    safetensors_weights_manager.get_slice(f"{export_prefix}self_attn.k_proj.weight"),
                    safetensors_weights_manager.get_slice(f"{export_prefix}self_attn.v_proj.weight"),
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                )
            )

            state_dict[f"{import_prefix}sequence_mixer.c_proj.weight"] = safetensors_weights_manager.get_tensor(
                f"{export_prefix}self_attn.o_proj.weight"
            )
        else:
            raise ValueError(f"unexpected sequence_mixer_type ({sequence_mixer_block_types[layer_idx]})")

    return state_dict


def _get_sequence_mixer_block_types(config: GPTBaseConfig) -> list[str]:
    blocks = getattr(config, "sequence_mixer_blocks")

    def _get(block, key):
        return block.get(key) if isinstance(block, dict) else getattr(block, key)

    seq_mixer_block_types = []
    for block in blocks:
        block_type = _get(block, "sequence_mixer_type")
        # block type mamba to use HybridMambaCache
        if block_type == "mamba2":
            block_type = "mamba"
        elif block_type == "softmax_attention":
            block_type = "attention"
        seq_mixer_block_types.append(block_type)

    return seq_mixer_block_types


def _export_granitemoehybrid_config(config: GPTBaseConfig) -> GraniteMoeHybridConfig:
    assert config.normalization_function == "rmsnorm"
    assert config.position_embedding_type == "nope"

    config.check_equal_for_all_and_get_value("mlp_blocks", "add_bias", False)
    config.check_equal_for_all_and_get_value("mlp_blocks", "activation_function", "swiglu")

    # Allow for 0 experts: if all mlp_blocks have mlp_type "None", set num_local_experts to 0
    mlp_types = [
        block["mlp_type"] if isinstance(block, dict) else getattr(block, "mlp_type") for block in config.mlp_blocks
    ]
    if all(t == "MLP" for t in mlp_types):
        num_local_experts = 0
        num_experts_per_tok = 0
        shared_intermediate_size = config.check_equal_for_all_and_get_value("mlp_blocks", "intermediate_size")
    else:
        num_local_experts = config.check_equal_for_all_and_get_value("mlp_blocks", "num_experts")
        num_experts_per_tok = config.check_equal_for_all_and_get_value("mlp_blocks", "num_experts_per_tok")
        shared_intermediate_size = config.check_equal_for_all_and_get_value("mlp_blocks", "shared_intermediate_size")

    original_config = GraniteMoeHybridConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.check_equal_for_all_and_get_value(
            key="sequence_mixer_blocks", key_block="num_attention_heads", sequence_mixer_type="softmax_attention"
        ),
        shared_intermediate_size=0 if shared_intermediate_size is None else shared_intermediate_size,
        num_key_value_heads=config.check_equal_for_all_and_get_value(
            key="sequence_mixer_blocks", key_block="num_key_value_heads", sequence_mixer_type="softmax_attention"
        ),
        intermediate_size=config.check_equal_for_all_and_get_value("mlp_blocks", "intermediate_size"),
        hidden_act="silu",
        rms_norm_eps=config.layer_norm_epsilon,
        use_cache=config.use_cache,
        attention_bias=config.check_equal_for_all_and_get_value(
            key="sequence_mixer_blocks", key_block="add_bias", sequence_mixer_type="softmax_attention"
        ),
        tie_word_embeddings=config.tie_word_embeddings,
        initializer_range=config.initializer_range,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
        attention_dropout=config.check_equal_for_all_and_get_value(
            key="sequence_mixer_blocks", key_block="dropout", sequence_mixer_type="softmax_attention"
        ),
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
        router_aux_loss_coef=config.router_aux_loss_coef,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        embedding_multiplier=1 if config.m_emb is None else config.m_emb,
        residual_multiplier=1 if config.m_residual is None else config.m_residual,
        logits_scaling=1 if config.m_width is None else config.m_width,
        attention_multiplier=config.check_equal_for_all_and_get_value(
            key="sequence_mixer_blocks", key_block="attention_multiplier", sequence_mixer_type="softmax_attention"
        ),
        mamba_expand=divide_if_divisible(
            config.check_equal_for_all_and_get_value(
                key="sequence_mixer_blocks", key_block="intermediate_size", sequence_mixer_type="mamba2"
            ),
            config.hidden_size,
            "",
        ),
        mamba_n_groups=config.check_equal_for_all_and_get_value(
            key="sequence_mixer_blocks", key_block="num_groups", sequence_mixer_type="mamba2"
        ),
        mamba_n_heads=config.check_equal_for_all_and_get_value(
            key="sequence_mixer_blocks", key_block="num_heads", sequence_mixer_type="mamba2"
        ),
        mamba_d_state=config.check_equal_for_all_and_get_value(
            key="sequence_mixer_blocks", key_block="state_size", sequence_mixer_type="mamba2"
        ),
        mamba_d_conv=config.check_equal_for_all_and_get_value(
            key="sequence_mixer_blocks", key_block="conv_kernel_size", sequence_mixer_type="mamba2"
        ),
        mamba_chunk_size=config.check_equal_for_all_and_get_value(
            key="sequence_mixer_blocks", key_block="chunk_size", sequence_mixer_type="mamba2"
        ),
        mamba_conv_bias=config.check_equal_for_all_and_get_value(
            key="sequence_mixer_blocks", key_block="use_conv_bias", sequence_mixer_type="mamba2"
        ),
        mamba_proj_bias=config.check_equal_for_all_and_get_value(
            key="sequence_mixer_blocks", key_block="add_bias", sequence_mixer_type="mamba2"
        ),
        layer_types=_get_sequence_mixer_block_types(config),
        # TODO drop normalization_function, position_embedding_type, init_method
        normalization_function=config.normalization_function,
        position_embedding_type=config.position_embedding_type,
        init_method=config.init_method,
        architectures=[GraniteMoeHybridForCausalLM.__name__],
    )

    return original_config


def _export_granitemoehybrid_state_dict(
    config: GPTBaseConfig, safetensors_weights_manager: SafeTensorsWeightsManager
) -> dict:
    num_attention_heads = config.check_equal_for_all_and_get_value(
        "sequence_mixer_blocks", "num_attention_heads", sequence_mixer_type="softmax_attention"
    )

    num_key_value_heads = config.check_equal_for_all_and_get_value(
        "sequence_mixer_blocks", "num_key_value_heads", sequence_mixer_type="softmax_attention"
    )

    sequence_mixer_block_types = _get_sequence_mixer_block_types(config)

    state_dict = {
        "model.embed_tokens.weight": safetensors_weights_manager.get_tensor("transformer.wte.weight"),
        "model.norm.weight": safetensors_weights_manager.get_tensor("transformer.ln_f.weight"),
    }

    if safetensors_weights_manager.has_tensor("lm_head.weight"):
        state_dict["lm_head.weight"] = safetensors_weights_manager.get_tensor("lm_head.weight")

    for layer_idx in range(config.num_layers):
        use_interleaved_weights = config.mlp_blocks[layer_idx].use_interleaved_weights
        import_prefix = f"transformer.h.{layer_idx}."
        export_prefix = f"model.layers.{layer_idx}."

        state_dict[f"{export_prefix}input_layernorm.weight"] = safetensors_weights_manager.get_tensor(
            f"{import_prefix}ln_1.weight"
        )
        state_dict[f"{export_prefix}post_attention_layernorm.weight"] = safetensors_weights_manager.get_tensor(
            f"{import_prefix}ln_2.weight"
        )

        if safetensors_weights_manager.has_tensor(f"{import_prefix}mlp_block.gate.weight"):
            state_dict[f"{export_prefix}block_sparse_moe.router.layer.weight"] = (
                safetensors_weights_manager.get_tensor(f"{import_prefix}mlp_block.gate.weight")
            )
            u, g = split_up_gate_tensor_for_mlp(
                safetensors_weights_manager.get_tensor(f"{import_prefix}mlp_block.c_fc.weight"),
                is_interleaved=use_interleaved_weights,
                dim=1,
            )
            state_dict[f"{export_prefix}block_sparse_moe.input_linear.weight"] = torch.cat([g, u], dim=1)
            state_dict[f"{export_prefix}block_sparse_moe.output_linear.weight"] = (
                safetensors_weights_manager.get_tensor(f"{import_prefix}mlp_block.c_proj.weight")
            )

            if safetensors_weights_manager.has_tensor(f"{import_prefix}mlp_block.c_fc_shared.weight"):
                u, g = split_up_gate_tensor_for_mlp(
                    safetensors_weights_manager.get_tensor(f"{import_prefix}mlp_block.c_fc_shared.weight"),
                    is_interleaved=config.mlp_blocks[layer_idx].use_interleaved_weights_for_shared_experts,
                )
                state_dict[f"{export_prefix}shared_mlp.input_linear.weight"] = torch.cat([g, u], dim=0)
                state_dict[f"{export_prefix}shared_mlp.output_linear.weight"] = safetensors_weights_manager.get_tensor(
                    f"{import_prefix}mlp_block.c_proj_shared.weight"
                )
        else:
            u, g = split_up_gate_tensor_for_mlp(
                safetensors_weights_manager.get_tensor(f"{import_prefix}mlp_block.c_fc.weight"),
                is_interleaved=use_interleaved_weights,
            )
            state_dict[f"{export_prefix}shared_mlp.input_linear.weight"] = torch.cat([g, u], dim=0)
            state_dict[f"{export_prefix}shared_mlp.output_linear.weight"] = safetensors_weights_manager.get_tensor(
                f"{import_prefix}mlp_block.c_proj.weight"
            )

        if sequence_mixer_block_types[layer_idx] == "mamba":
            state_dict[f"{export_prefix}mamba.conv1d.weight"] = safetensors_weights_manager.get_tensor(
                f"{import_prefix}sequence_mixer.conv1d.weight"
            )
            if safetensors_weights_manager.has_tensor(f"{import_prefix}sequence_mixer.conv1d.bias"):
                state_dict[f"{export_prefix}mamba.conv1d.bias"] = safetensors_weights_manager.get_tensor(
                    f"{import_prefix}sequence_mixer.conv1d.bias"
                )
            state_dict[f"{export_prefix}mamba.in_proj.weight"] = safetensors_weights_manager.get_tensor(
                f"{import_prefix}sequence_mixer.in_proj.weight"
            )
            if safetensors_weights_manager.has_tensor(f"{import_prefix}sequence_mixer.in_proj.bias"):
                state_dict[f"{export_prefix}mamba.in_proj.bias"] = safetensors_weights_manager.get_tensor(
                    f"{import_prefix}sequence_mixer.in_proj.bias"
                )
            state_dict[f"{export_prefix}mamba.dt_bias"] = safetensors_weights_manager.get_tensor(
                f"{import_prefix}sequence_mixer.decay_gate.dt_bias"
            )
            state_dict[f"{export_prefix}mamba.A_log"] = safetensors_weights_manager.get_tensor(
                f"{import_prefix}sequence_mixer.decay_gate.A_log"
            )
            state_dict[f"{export_prefix}mamba.D"] = safetensors_weights_manager.get_tensor(
                f"{import_prefix}sequence_mixer.D"
            )
            state_dict[f"{export_prefix}mamba.out_proj.weight"] = safetensors_weights_manager.get_tensor(
                f"{import_prefix}sequence_mixer.out_proj.weight"
            )
            if safetensors_weights_manager.has_tensor(f"{import_prefix}sequence_mixer.out_proj.bias"):
                state_dict[f"{export_prefix}mamba.out_proj.bias"] = safetensors_weights_manager.get_tensor(
                    f"{import_prefix}sequence_mixer.out_proj.bias"
                )
            state_dict[f"{export_prefix}mamba.norm.weight"] = safetensors_weights_manager.get_tensor(
                f"{import_prefix}sequence_mixer.norm.weight"
            )
        elif sequence_mixer_block_types[layer_idx] == "attention":
            query_weight, key_weight, value_weight = split_query_key_value_tensor_for_attention(
                safetensors_weights_manager.get_tensor(f"{import_prefix}sequence_mixer.c_attn.weight"),
                num_attention_heads,
                num_key_value_heads,
            )
            state_dict[f"{export_prefix}self_attn.q_proj.weight"] = query_weight
            state_dict[f"{export_prefix}self_attn.k_proj.weight"] = key_weight
            state_dict[f"{export_prefix}self_attn.v_proj.weight"] = value_weight

            state_dict[f"{export_prefix}self_attn.o_proj.weight"] = safetensors_weights_manager.get_tensor(
                f"{import_prefix}sequence_mixer.c_proj.weight"
            )
        else:
            raise ValueError(f"unexpected sequence_mixer_type ({sequence_mixer_block_types[layer_idx]})")

    return state_dict
