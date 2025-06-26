# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    GraniteMoeSharedConfig,
    GraniteMoeSharedForCausalLM,
)

from ...utils import SafeTensorsWeightsManager, download_repo
from ..modeling_utils import (
    get_attention_head_type,
    interleave_query_key_value_tensor_for_attention,
    split_query_key_value_tensor_for_attention,
)
from ..models import GPTBaseConfig


def import_from_huggingface_granitemoeshared(pretrained_model_name_or_path: str, save_path: str) -> None:
    original_config, tokenizer, downloaded_model_path = download_repo(pretrained_model_name_or_path)
    config = _import_config_from_huggingface(original_config)
    num_attention_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads")

    safetensors_weights_manager = SafeTensorsWeightsManager(downloaded_model_path)
    state_dict = _import_state_dict_from_huggingface(
        safetensors_weights_manager,
        config.num_layers,
        num_attention_heads,
        config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_key_value_heads"),
        config.hidden_size // num_attention_heads,
    )

    SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)
    config.save_pretrained(save_path)

    generation_config = GenerationConfig.from_model_config(config)
    generation_config.save_pretrained(save_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path, legacy_format=False)


def _import_config_from_huggingface(original_config: GraniteMoeSharedConfig) -> GPTBaseConfig:
    assert original_config.hidden_act == "silu"
    assert not original_config.attention_bias

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
        m_emb=None if original_config.embedding_multiplier == 1 else original_config.embedding_multiplier,
        m_residual=None if original_config.residual_multiplier == 1 else original_config.residual_multiplier,
        m_width=None if original_config.logits_scaling == 1 else original_config.logits_scaling,
        sequence_mixer_blocks=[
            {
                "sequence_mixer_type": "softmax_attention",
                "num_attention_heads": original_config.num_attention_heads,
                "num_key_value_heads": original_config.num_key_value_heads,
                "attention_multiplier": original_config.attention_multiplier,
                "add_bias": False,
                "softmax_dropout": original_config.attention_dropout,
            }
            for _ in range(original_config.num_hidden_layers)
        ],
        mlp_blocks=[
            {
                "mlp_type": "MoE",
                "intermediate_size": original_config.intermediate_size,
                "shared_intermediate_size": (
                    None if original_config.shared_intermediate_size == 0 else original_config.shared_intermediate_size
                ),
                "num_experts": original_config.num_local_experts,
                "num_experts_per_tok": original_config.num_experts_per_tok,
                "activation_function": "swiglu",
                "add_bias": False,
            }
            for _ in range(original_config.num_hidden_layers)
        ],
    )

    return config


def _import_state_dict_from_huggingface(
    safetensors_weights_manager: SafeTensorsWeightsManager,
    num_layers: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> None:
    attention_head_type = get_attention_head_type(num_heads, num_key_value_heads)

    state_dict = {
        "transformer.wte.weight": safetensors_weights_manager.get_tensor("model.embed_tokens.weight"),
        "transformer.ln_f.weight": safetensors_weights_manager.get_tensor("model.norm.weight"),
    }

    if safetensors_weights_manager.has_tensor("lm_head.weight"):
        state_dict["lm_head.weight"] = safetensors_weights_manager.get_tensor("lm_head.weight")

    for layer_idx in range(num_layers):
        state_dict[f"transformer.h.{layer_idx}.ln_1.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.input_layernorm.weight"
        )
        state_dict[f"transformer.h.{layer_idx}.ln_2.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        )

        state_dict[f"transformer.h.{layer_idx}.mlp_block.gate.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.block_sparse_moe.router.layer.weight"
        )

        state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc.weight"] = _split_and_reorder_for_glu(
            safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.block_sparse_moe.input_linear.weight"),
            dim=1,
        )
        state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.block_sparse_moe.output_linear.weight"
        )

        if safetensors_weights_manager.has_tensor(f"model.layers.{layer_idx}.shared_mlp.input_linear.weight"):
            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc_shared.weight"] = _split_and_reorder_for_glu(
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.shared_mlp.input_linear.weight"),
                dim=0,
            )
            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj_shared.weight"] = (
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.shared_mlp.output_linear.weight")
            )

        state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_attn.weight"] = (
            interleave_query_key_value_tensor_for_attention(
                safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.q_proj.weight"),
                safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.k_proj.weight"),
                safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.v_proj.weight"),
                num_heads,
                num_key_value_heads,
                head_dim,
                attention_head_type,
            )
        )
        state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        )

    return state_dict


def export_to_huggingface_granitemoeshared(pretrained_model_name_or_path: str, save_path: str) -> None:
    config: GPTBaseConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    original_config = _export_config_to_huggingface(config)
    num_attention_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads")

    safetensors_weights_manager = SafeTensorsWeightsManager(pretrained_model_name_or_path)
    state_dict = _export_state_dict_to_huggingface(
        safetensors_weights_manager,
        config.num_layers,
        num_attention_heads,
        config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_key_value_heads"),
        config.hidden_size // num_attention_heads,
    )

    SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)
    original_config.save_pretrained(save_path)

    original_generation_config = GenerationConfig.from_model_config(original_config)
    original_generation_config.save_pretrained(save_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        tokenizer.save_pretrained(save_path, legacy_format=False)
    except:
        pass


def _export_config_to_huggingface(config: GPTBaseConfig) -> GraniteMoeSharedConfig:
    assert config.normalization_function == "rmsnorm"
    assert config.position_embedding_type == "rope"

    config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "add_bias", False)
    config.check_equal_for_all_and_get_value("mlp_blocks", "add_bias", False)
    config.check_equal_for_all_and_get_value("mlp_blocks", "activation_function", "swiglu")
    config.check_equal_for_all_and_get_value("mlp_blocks", "mlp_type", "MoE")
    shared_intermediate_size = config.check_equal_for_all_and_get_value("mlp_blocks", "shared_intermediate_size")

    original_config = GraniteMoeSharedConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads"),
        shared_intermediate_size=0 if shared_intermediate_size is None else shared_intermediate_size,
        num_key_value_heads=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_key_value_heads"),
        intermediate_size=config.check_equal_for_all_and_get_value("mlp_blocks", "intermediate_size"),
        hidden_act="silu",
        rms_norm_eps=config.layer_norm_epsilon,
        use_cache=config.use_cache,
        attention_bias=False,
        tie_word_embeddings=config.tie_word_embeddings,
        initializer_range=config.initializer_range,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
        attention_dropout=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "softmax_dropout"),
        num_local_experts=config.check_equal_for_all_and_get_value("mlp_blocks", "num_experts"),
        num_experts_per_tok=config.check_equal_for_all_and_get_value("mlp_blocks", "num_experts_per_tok"),
        router_aux_loss_coef=config.router_aux_loss_coef,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        embedding_multiplier=1 if config.m_emb is None else config.m_emb,
        residual_multiplier=1 if config.m_residual is None else config.m_residual,
        logits_scaling=1 if config.m_width is None else config.m_width,
        attention_multiplier=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "attention_multiplier"),
        architectures=[GraniteMoeSharedForCausalLM.__name__],
    )

    return original_config


def _export_state_dict_to_huggingface(
    safetensors_weights_manager: SafeTensorsWeightsManager,
    num_layers: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> None:
    attention_head_type = get_attention_head_type(num_heads, num_key_value_heads)

    state_dict = {
        "model.embed_tokens.weight": safetensors_weights_manager.get_tensor("transformer.wte.weight"),
        "model.norm.weight": safetensors_weights_manager.get_tensor("transformer.ln_f.weight"),
    }

    if safetensors_weights_manager.has_tensor("lm_head.weight"):
        state_dict["lm_head.weight"] = safetensors_weights_manager.get_tensor("lm_head.weight")

    for layer_idx in range(num_layers):
        state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = safetensors_weights_manager.get_tensor(
            f"transformer.h.{layer_idx}.ln_1.weight"
        )
        state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = (
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.ln_2.weight")
        )

        state_dict[f"model.layers.{layer_idx}.block_sparse_moe.router.layer.weight"] = (
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.gate.weight")
        )

        state_dict[f"model.layers.{layer_idx}.block_sparse_moe.input_linear.weight"] = _split_and_reorder_for_glu(
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc.weight"), dim=1
        )
        state_dict[f"model.layers.{layer_idx}.block_sparse_moe.output_linear.weight"] = (
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_proj.weight")
        )

        if safetensors_weights_manager.has_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc_shared.weight"):
            state_dict[f"model.layers.{layer_idx}.shared_mlp.input_linear.weight"] = _split_and_reorder_for_glu(
                safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc_shared.weight"),
                dim=0,
            )
            state_dict[f"model.layers.{layer_idx}.shared_mlp.output_linear.weight"] = (
                safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_proj_shared.weight")
            )

        query_weight, key_weight, value_weight = split_query_key_value_tensor_for_attention(
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.sequence_mixer.c_attn.weight"),
            num_heads,
            num_key_value_heads,
            head_dim,
            attention_head_type,
        )
        state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = query_weight
        state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = key_weight
        state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = value_weight

        state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = safetensors_weights_manager.get_tensor(
            f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"
        )

    return state_dict


def _split_and_reorder_for_glu(weight: torch.Tensor, dim: int) -> torch.Tensor:
    x, y = weight.chunk(2, dim=dim)
    weight = torch.cat([y, x], dim=dim)
    return weight
