# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import AutoConfig, AutoTokenizer, GenerationConfig, LlamaConfig, LlamaForCausalLM

from ...utils import SafeTensorsWeightsManager, download_repo
from ..modeling_utils import (
    get_attention_head_type,
    interleave_query_key_value_tensor_for_attention,
    interleave_up_gate_tensor_for_mlp,
    split_query_key_value_tensor_for_attention,
    split_up_gate_tensor_for_mlp,
)
from ..models import GPTBaseConfig


def import_from_huggingface_llama(pretrained_model_name_or_path: str, save_path: str) -> None:
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


def _import_config_from_huggingface(original_config: LlamaConfig) -> GPTBaseConfig:
    assert original_config.hidden_act == "silu"
    assert original_config.mlp_bias == original_config.attention_bias

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
        bos_token_id=original_config.bos_token_id,
        eos_token_id=original_config.eos_token_id,
        pad_token_id=original_config.pad_token_id,
        sequence_mixer_blocks=[
            {
                "sequence_mixer_type": "softmax_attention",
                "add_bias": original_config.attention_bias,
                "num_attention_heads": original_config.num_attention_heads,
                "num_key_value_heads": original_config.num_key_value_heads,
                "softmax_dropout": original_config.attention_dropout,
            }
            for _ in range(original_config.num_hidden_layers)
        ],
        mlp_blocks=[
            {
                "mlp_type": "MLP",
                "add_bias": original_config.mlp_bias,
                "activation_function": "swiglu",
                "intermediate_size": original_config.intermediate_size,
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

        state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc.weight"] = interleave_up_gate_tensor_for_mlp(
            safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mlp.up_proj.weight"),
            safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mlp.gate_proj.weight"),
        )
        if f"model.layers.{layer_idx}.mlp.up_proj.bias" in safetensors_weights_manager:
            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc.bias"] = interleave_up_gate_tensor_for_mlp(
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mlp.up_proj.bias"),
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mlp.gate_proj.bias"),
            )

        state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.mlp.down_proj.weight"
        )
        if f"model.layers.{layer_idx}.mlp.down_proj.bias" in safetensors_weights_manager:
            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj.bias"] = safetensors_weights_manager.get_tensor(
                f"model.layers.{layer_idx}.mlp.down_proj.bias"
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
        if f"model.layers.{layer_idx}.self_attn.q_proj.bias" in safetensors_weights_manager:
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_attn.bias"] = (
                interleave_query_key_value_tensor_for_attention(
                    safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.q_proj.bias"),
                    safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.k_proj.bias"),
                    safetensors_weights_manager.get_slice(f"model.layers.{layer_idx}.self_attn.v_proj.bias"),
                    num_heads,
                    num_key_value_heads,
                    head_dim,
                    attention_head_type,
                )
            )

        state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"] = safetensors_weights_manager.get_tensor(
            f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        )
        if f"model.layers.{layer_idx}.self_attn.o_proj.bias" in safetensors_weights_manager:
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_proj.bias"] = (
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.self_attn.o_proj.bias")
            )

    return state_dict


def export_to_huggingface_llama(pretrained_model_name_or_path: str, save_path: str) -> None:
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


def _export_config_to_huggingface(config: GPTBaseConfig) -> LlamaConfig:
    assert config.normalization_function == "rmsnorm"
    assert config.position_embedding_type == "rope"
    assert config.m_emb is None
    assert config.m_residual is None
    assert config.m_width is None
    assert config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "attention_multiplier") is None

    config.check_equal_for_all_and_get_value("mlp_blocks", "activation_function", "swiglu")

    original_config = LlamaConfig(
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
        attention_bias=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "add_bias"),
        tie_word_embeddings=config.tie_word_embeddings,
        initializer_range=config.initializer_range,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
        attention_dropout=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "softmax_dropout"),
        mlp_bias=config.check_equal_for_all_and_get_value("mlp_blocks", "add_bias"),
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        architectures=[LlamaForCausalLM.__name__],
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

        up_weight, gate_weight = split_up_gate_tensor_for_mlp(
            safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc.weight")
        )
        state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_weight
        state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_weight

        if f"transformer.h.{layer_idx}.mlp_block.c_fc.bias" in safetensors_weights_manager:
            up_bias, gate_bias = split_up_gate_tensor_for_mlp(
                safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.mlp_block.c_fc.bias")
            )
            state_dict[f"model.layers.{layer_idx}.mlp.up_proj.bias"] = up_bias
            state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.bias"] = gate_bias

        state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = safetensors_weights_manager.get_tensor(
            f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"
        )
        if f"transformer.h.{layer_idx}.mlp_block.c_proj.bias" in safetensors_weights_manager:
            state_dict[f"model.layers.{layer_idx}.mlp.down_proj.bias"] = safetensors_weights_manager.get_tensor(
                f"transformer.h.{layer_idx}.mlp_block.c_proj.bias"
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

        if f"transformer.h.{layer_idx}.sequence_mixer.c_attn.bias" in safetensors_weights_manager:
            query_bias, key_bias, value_bias = split_query_key_value_tensor_for_attention(
                safetensors_weights_manager.get_tensor(f"transformer.h.{layer_idx}.sequence_mixer.c_attn.bias"),
                num_heads,
                num_key_value_heads,
                head_dim,
                attention_head_type,
            )
            state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.bias"] = query_bias
            state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.bias"] = key_bias
            state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.bias"] = value_bias

        state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = safetensors_weights_manager.get_tensor(
            f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"
        )
        if f"transformer.h.{layer_idx}.sequence_mixer.c_proj.bias" in safetensors_weights_manager:
            state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.bias"] = safetensors_weights_manager.get_tensor(
                f"transformer.h.{layer_idx}.sequence_mixer.c_proj.bias"
            )

    return state_dict
