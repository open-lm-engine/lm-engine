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


def _import_qwen2_moe_config(original_config: Qwen2MoeConfig) -> GPTBaseConfig:
    assert original_config.hidden_act == "silu"

    mlp_blocks = []
    for layer_idx in range(original_config.num_hidden_layers):
        if (layer_idx not in original_config.mlp_only_layers) and (
            original_config.num_experts > 0 and (layer_idx + 1) % original_config.decoder_sparse_step == 0
        ):
            mlp_block = {
                "mlp_type": "MoE",
                "intermediate_size": original_config.moe_intermediate_size,
                "shared_intermediate_size": original_config.shared_expert_intermediate_size,
                "shared_expert_gating": True,
                "num_experts": original_config.num_experts,
                "num_experts_per_tok": original_config.num_experts_per_tok,
                "activation_function": "swiglu",
                "add_bias": False,
                "normalized_topk": original_config.norm_topk_prob,
            }
        else:
            mlp_block = {
                "mlp_type": "MLP",
                "intermediate_size": original_config.intermediate_size,
                "activation_function": "swiglu",
                "add_bias": False,
            }

        mlp_blocks.append(mlp_block)

    sequence_mixer_blocks = []
    for layer_idx in range(original_config.num_hidden_layers):
        sliding_window = None
        if original_config.use_sliding_window and layer_idx >= original_config.max_window_layers:
            sliding_window = original_config.sliding_window

        sequence_mixer_block = {
            "sequence_mixer_type": "softmax_attention",
            "num_attention_heads": original_config.num_attention_heads,
            "num_key_value_heads": original_config.num_key_value_heads,
            "add_bias": False,
            "sliding_window": sliding_window,
            "qkv_bias": original_config.qkv_bias,
            "softmax_dropout": original_config.attention_dropout,
        }

        sequence_mixer_blocks.append(sequence_mixer_block)

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
        sequence_mixer_blocks=sequence_mixer_blocks,
        mlp_blocks=mlp_blocks,
    )

    return config


def _import_qwen2_moe_state_dict(
    config: GPTBaseConfig, safetensors_weights_manager: SafeTensorsWeightsManager
) -> dict:
    num_attention_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads")
    num_key_value_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_key_value_heads")
    qkv_bias = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "qkv_bias")
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

        # MoE
        if safetensors_weights_manager.has_tensor(f"model.layers.{layer_idx}.mlp.gate.weight"):
            state_dict[f"transformer.h.{layer_idx}.mlp_block.gate.weight"] = safetensors_weights_manager.get_tensor(
                f"model.layers.{layer_idx}.mlp.gate.weight"
            )

            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc.weight"] = torch.stack(
                [
                    interleave_up_gate_tensor_for_mlp(
                        safetensors_weights_manager.get_tensor(
                            f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
                        ),
                        safetensors_weights_manager.get_tensor(
                            f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
                        ),
                    )
                    for expert_idx in range(config.mlp_blocks[layer_idx].num_experts)
                ]
            )

            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"] = torch.stack(
                [
                    safetensors_weights_manager.get_tensor(
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
                    )
                    for expert_idx in range(config.mlp_blocks[layer_idx].num_experts)
                ]
            )

            if safetensors_weights_manager.has_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight"):
                state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc_shared.weight"] = (
                    interleave_up_gate_tensor_for_mlp(
                        safetensors_weights_manager.get_tensor(
                            f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight"
                        ),
                        safetensors_weights_manager.get_tensor(
                            f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight"
                        ),
                    )
                )

                state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj_shared.weight"] = (
                    safetensors_weights_manager.get_tensor(
                        f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight"
                    )
                )

                state_dict[f"transformer.h.{layer_idx}.mlp_block.shared_expert_gate.weight"] = (
                    safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight")
                )
        # MLP
        else:
            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc.weight"] = interleave_up_gate_tensor_for_mlp(
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mlp.up_proj.weight"),
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mlp.gate_proj.weight"),
            )

            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"] = safetensors_weights_manager.get_tensor(
                f"model.layers.{layer_idx}.mlp.down_proj.weight"
            )

        keys = ["weight"] + (["bias"] if qkv_bias else [])
        for key in keys:
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_attn.{key}"] = (
                interleave_query_key_value_tensor_for_attention(
                    safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.q_proj.{key}"),
                    safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.k_proj.{key}"),
                    safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.v_proj.{key}"),
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

    mlp_only_layers = [
        layer_idx for layer_idx, mlp_block in enumerate(config.mlp_blocks) if mlp_block.mlp_type == "MLP"
    ]

    max_window_layers = None
    use_sliding_window = False
    for layer_idx in range(config.num_layers):
        block = config.sequence_mixer_blocks[layer_idx]
        if config.sequence_mixer_blocks[layer_idx]:
            use_sliding_window = use_sliding_window or block.sliding_window is not None
            if max_window_layers is None and use_sliding_window:
                max_window_layers = layer_idx

    original_config = Qwen2MoeConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads"),
        num_key_value_heads=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_key_value_heads"),
        intermediate_size=config.check_equal_for_all_and_get_value("mlp_blocks", "intermediate_size", mlp_type="MLP"),
        moe_intermediate_size=config.check_equal_for_all_and_get_value(
            "mlp_blocks", "intermediate_size", mlp_type="MoE"
        ),
        shared_expert_intermediate_size=config.check_equal_for_all_and_get_value(
            "mlp_blocks", "shared_intermediate_size", mlp_type="MoE"
        ),
        hidden_act="silu",
        rms_norm_eps=config.layer_norm_epsilon,
        use_cache=config.use_cache,
        use_sliding_window=use_sliding_window,
        max_window_layers=max_window_layers,
        tie_word_embeddings=config.tie_word_embeddings,
        initializer_range=config.initializer_range,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
        attention_dropout=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "softmax_dropout"),
        num_experts=config.check_equal_for_all_and_get_value("mlp_blocks", "num_experts", mlp_type="MoE"),
        num_experts_per_tok=config.check_equal_for_all_and_get_value(
            "mlp_blocks", "num_experts_per_tok", mlp_type="MoE"
        ),
        router_aux_loss_coef=config.router_aux_loss_coef,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        qkv_bias=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "qkv_bias"),
        mlp_only_layers=mlp_only_layers,
        norm_topk_prob=config.check_equal_for_all_and_get_value("mlp_blocks", "normalized_topk", mlp_type="MoE"),
        architectures=[Qwen2MoeForCausalLM.__name__],
    )

    return original_config


def _export_qwen2_moe_state_dict(
    config: GPTBaseConfig, safetensors_weights_manager: SafeTensorsWeightsManager
) -> dict:
    num_attention_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads")
    num_key_value_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_key_value_heads")
    qkv_bias = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "qkv_bias")

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

        # MoE layer
        if safetensors_weights_manager.has_tensor(f"transformer.h.{layer_idx}.mlp_block.gate.weight"):
            state_dict[f"model.layers.{layer_idx}.mlp.gate.weight"] = safetensors_weights_manager.get_tensor(
                f"transformer.h.{layer_idx}.mlp_block.gate.weight"
            )

            for expert_idx in range(config.mlp_blocks[layer_idx].num_experts):
                up_weight, gate_weight = split_up_gate_tensor_for_mlp(
                    safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc.weight")[
                        expert_idx
                    ]
                )

                state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = up_weight
                state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = gate_weight

                state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = (
                    safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_proj.weight")[
                        expert_idx
                    ]
                )

            if safetensors_weights_manager.has_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc_shared.weight"):
                up_weight, gate_weight = split_up_gate_tensor_for_mlp(
                    safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc_shared.weight")
                )

                state_dict[f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight"] = gate_weight
                state_dict[f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight"] = up_weight
                state_dict[f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight"] = (
                    safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_proj_shared.weight")
                )

                state_dict[f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight"] = (
                    safetensors_weights_manager.get_tensor(
                        f"transformer.h.{layer_idx}.mlp_block.shared_expert_gate.weight"
                    )
                )
        # MLP layer
        else:
            up_weight, gate_weight = split_up_gate_tensor_for_mlp(
                safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc.weight")
            )

            state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_weight
            state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_weight

            state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = safetensors_weights_manager.get_tensor(
                f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"
            )

        keys = ["weight"] + (["bias"] if qkv_bias else [])
        for key in keys:
            query_weight, key_weight, value_weight = split_query_key_value_tensor_for_attention(
                safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.sequence_mixer.c_attn.{key}"),
                num_attention_heads,
                num_key_value_heads,
            )
            state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.{key}"] = query_weight
            state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.{key}"] = key_weight
            state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.{key}"] = value_weight

        state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = safetensors_weights_manager.get_tensor(
            f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"
        )

    return state_dict
