# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from ...utils import SafeTensorsWeightsManager, divide_if_divisible
from ..modeling_utils import (
    interleave_query_key_value_tensor_for_attention,
    interleave_up_gate_tensor_for_mlp,
    split_query_key_value_tensor_for_attention,
    split_up_gate_tensor_for_mlp,
)
from ..models import GPTBaseConfig
from .granitemoeshared import _split_and_reorder_for_glu


def _import_qwen2_moe_config(original_config: Qwen2MoeConfig) -> GPTBaseConfig:
    assert original_config.hidden_act == "silu"

    config = GPTBaseConfig(
        vocab_size=original_config.vocab_size,
        max_position_embeddings=original_config.max_position_embeddings,
        hidden_size=original_config.hidden_size,
        num_layers=original_config.num_hidden_layers,
        position_embedding_type="rope",
        normalization_function="rmsnorm",
        layer_norm_epsilon=original_config.rms_norm_eps,
        use_cache=original_config.use_cache,
        tie_word_embeddings=original_config.tie_word_embeddings,
        initializer_range=original_config.initializer_range,
        rope_theta=original_config.rope_theta,
        rope_scaling=original_config.rope_scaling,
        router_aux_loss_coef=original_config.router_aux_loss_coef,
        bos_token_id=original_config.bos_token_id,
        eos_token_id=original_config.eos_token_id,
        pad_token_id=original_config.pad_token_id,
        sequence_mixer_blocks=[
            {
                "sequence_mixer_type": "softmax_attention",
                "num_attention_heads": original_config.num_attention_heads,
                "num_key_value_heads": original_config.num_key_value_heads,
                "add_bias": False,
                "softmax_dropout": original_config.attention_dropout,
            }
            for _ in range(original_config.num_hidden_layers)
        ],
        mlp_blocks=[
            {
                "mlp_type": "MoE",
                "intermediate_size": original_config.intermediate_size,
                "num_experts": original_config.num_experts,
                "num_experts_per_tok": original_config.num_experts_per_tok,
                "activation_function": "swiglu",
                "add_bias": False,
            }
            for _ in range(original_config.num_hidden_layers)
        ],
    )

    return config


def _import_qwen2_moe_state_dict(
    config: GPTBaseConfig, safetensors_weights_manager: SafeTensorsWeightsManager
) -> dict:
    num_attention_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads")
    num_key_value_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_key_value_heads")
    head_dim = divide_if_divisible(config.hidden_size, num_attention_heads, "")

    state_dict = {
        "transformer.wte.weight": safetensors_weights_manager.get_tensor("model.embed_tokens.weight"),
        "transformer.ln_f.weight": safetensors_weights_manager.get_tensor("model.norm.weight"),
    }

    if safetensors_weights_manager.has_tensor("lm_head.weight"):
        state_dict["lm_head.weight"] = safetensors_weights_manager.get_tensor("lm_head.weight")

    for layer_idx in range(config.num_layers):
        state_dict[f"transformer.h.{layer_idx}.ln_1.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.input_layernorm.weight"
        )
        state_dict[f"transformer.h.{layer_idx}.ln_2.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        )

        state_dict[f"transformer.h.{layer_idx}.mlp_block.gate.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.mlp.gate.weight"
        )

        c_fc_weights = []
        down_weights = []
        for expert_idx in range(config.mlp_blocks[layer_idx].num_experts):
            c_fc_weights.append(
                interleave_up_gate_tensor_for_mlp(
                    safetensors_weights_manager.get_tensor(
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
                    ),
                    safetensors_weights_manager.get_tensor(
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
                    ),
                )
            )

            down_weights.append(
                safetensors_weights_manager.get_tensor(
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
                )
            )

        state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc.weight"] = torch.stack(c_fc_weights)

        state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"] = torch.stack(down_weights)

        if safetensors_weights_manager.has_tensor(f"model.layers.{layer_idx}.shared_expert.gate_proj.weight"):
            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc_shared.weight"] = interleave_up_gate_tensor_for_mlp(
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight"),
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight"),
            )

            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj_shared.weight"] = (
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.shared_expert.down_proj.weight")
            )

        state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_attn.weight"] = (
            interleave_query_key_value_tensor_for_attention(
                safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.q_proj.weight"),
                safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.k_proj.weight"),
                safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.v_proj.weight"),
                num_attention_heads,
                num_key_value_heads,
                head_dim,
            )
        )
        state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        )

    return state_dict


def _export_qwen2_moe_config(config: GPTBaseConfig) -> Qwen2MoeConfig:
    assert config.normalization_function == "rmsnorm"
    assert config.position_embedding_type == "rope"

    config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "add_bias", False)
    config.check_equal_for_all_and_get_value("mlp_blocks", "add_bias", False)
    config.check_equal_for_all_and_get_value("mlp_blocks", "activation_function", "swiglu")
    config.check_equal_for_all_and_get_value("mlp_blocks", "mlp_type", "MoE")

    original_config = Qwen2MoeConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads"),
        num_key_value_heads=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_key_value_heads"),
        intermediate_size=config.check_equal_for_all_and_get_value("mlp_blocks", "intermediate_size"),
        hidden_act="silu",
        rms_norm_eps=config.layer_norm_epsilon,
        use_cache=config.use_cache,
        tie_word_embeddings=config.tie_word_embeddings,
        initializer_range=config.initializer_range,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
        attention_dropout=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "softmax_dropout"),
        num_experts=config.check_equal_for_all_and_get_value("mlp_blocks", "num_experts"),
        num_experts_per_tok=config.check_equal_for_all_and_get_value("mlp_blocks", "num_experts_per_tok"),
        router_aux_loss_coef=config.router_aux_loss_coef,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        qkv_bias=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "qkv_bias"),
        mlp_only_layers=None,
        architectures=[Qwen2MoeForCausalLM.__name__],
    )

    return original_config


def _export_qwen2_moe_state_dict(
    config: GPTBaseConfig, safetensors_weights_manager: SafeTensorsWeightsManager
) -> dict:
    num_attention_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads")
    num_key_value_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_key_value_heads")

    state_dict = {
        "model.embed_tokens.weight": safetensors_weights_manager.get_tensor("transformer.wte.weight"),
        "model.norm.weight": safetensors_weights_manager.get_tensor("transformer.ln_f.weight"),
    }

    if safetensors_weights_manager.has_tensor("lm_head.weight"):
        state_dict["lm_head.weight"] = safetensors_weights_manager.get_tensor("lm_head.weight")

    for layer_idx in range(config.num_layers):
        state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = safetensors_weights_manager.get_tensor(
            f"transformer.h.{layer_idx}.ln_1.weight"
        )
        state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = (
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.ln_2.weight")
        )

        state_dict[f"model.layers.{layer_idx}.mlp.gate.weight"] = safetensors_weights_manager.get_tensor(
            f"transformer.h.{layer_idx}.mlp_block.gate.weight"
        )

        for expert_idx in range(config.mlp_blocks[layer_idx].num_experts):
            up_weight, gate_weight = split_up_gate_tensor_for_mlp(
                safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc.weight")[expert_idx]
            )

            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = up_weight
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = gate_weight

            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = (
                safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_proj.weight")[
                    expert_idx
                ]
            )

        print(safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc.weight").size())

        if safetensors_weights_manager.has_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc_shared.weight"):
            state_dict[f"model.layers.{layer_idx}.shared_mlp.input_linear.weight"] = _split_and_reorder_for_glu(
                safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc_shared.weight"),
                dim=0,
            )
            state_dict[f"model.layers.{layer_idx}.shared_mlp.down_proj.weight"] = (
                safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_proj_shared.weight")
            )

        query_weight, key_weight, value_weight = split_query_key_value_tensor_for_attention(
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.sequence_mixer.c_attn.weight"),
            num_attention_heads,
            num_key_value_heads,
        )
        state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = query_weight
        state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = key_weight
        state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = value_weight

        state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = safetensors_weights_manager.get_tensor(
            f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"
        )

    return state_dict
