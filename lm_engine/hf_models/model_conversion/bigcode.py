# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import AutoConfig, AutoTokenizer, GenerationConfig, GPTBigCodeConfig, GPTBigCodeForCausalLM

from ...utils import SafeTensorsWeightsManager, divide_if_divisible, download_repo
from ..modeling_utils import get_attention_head_type
from ..models import GPTBaseConfig


def import_from_huggingface_bigcode(pretrained_model_name_or_path: str, save_path: str) -> None:
    original_config, tokenizer, downloaded_model_path = download_repo(pretrained_model_name_or_path)
    config = _import_config_from_huggingface(original_config)

    safetensors_weights_manager = SafeTensorsWeightsManager(downloaded_model_path)
    state_dict = _import_state_dict_from_huggingface(safetensors_weights_manager, config.num_layers)

    SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)
    config.save_pretrained(save_path)

    generation_config = GenerationConfig.from_model_config(config)
    generation_config.save_pretrained(save_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path, legacy_format=False)


def _import_config_from_huggingface(original_config: GPTBigCodeConfig) -> GPTBaseConfig:
    assert original_config.activation_function in ["gelu_pytorch_tanh", "gelu"]

    config = GPTBaseConfig(
        vocab_size=original_config.vocab_size,
        max_position_embeddings=original_config.n_positions,
        hidden_size=original_config.n_embd,
        num_layers=original_config.n_layer,
        position_embedding_type="learned_absolute",
        normalization_function="layernorm",
        layer_norm_epsilon=original_config.layer_norm_epsilon,
        use_cache=original_config.use_cache,
        tie_word_embeddings=original_config.tie_word_embeddings,
        initializer_range=original_config.initializer_range,
        bos_token_id=original_config.bos_token_id,
        eos_token_id=original_config.eos_token_id,
        pad_token_id=original_config.pad_token_id,
        sequence_mixer_blocks=[
            {
                "sequence_mixer_type": "softmax_attention",
                "add_bias": True,
                "num_attention_heads": original_config.n_head,
                "num_key_value_heads": 1 if original_config.multi_query else original_config.n_head,
            }
            for _ in range(original_config.n_layer)
        ],
        mlp_blocks=[
            {
                "mlp_type": "MLP",
                "activation_function": original_config.activation_function,
                "intermediate_size": original_config.n_inner,
                "add_bias": True,
            }
            for _ in range(original_config.n_layer)
        ],
    )

    return config


def _import_state_dict_from_huggingface(
    safetensors_weights_manager: SafeTensorsWeightsManager, num_layers: int
) -> None:
    state_dict = {key: safetensors_weights_manager.get_tensor(key) for key in safetensors_weights_manager}

    for layer_idx in range(num_layers):
        # sequence_mixer.c_attn
        state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_attn.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.attn.c_attn.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.attn.c_attn.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_attn.bias"] = bias

        # sequence_mixer.c_proj
        state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.attn.c_proj.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.attn.c_proj.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_proj.bias"] = bias

        # mlp_block.c_fc
        state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.mlp.c_fc.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.mlp.c_fc.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc.bias"] = bias

        # mlp_block.c_proj
        state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.mlp.c_proj.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.mlp.c_proj.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj.bias"] = bias

    return state_dict


def export_to_huggingface_bigcode(pretrained_model_name_or_path: str, save_path: str) -> None:
    config: GPTBaseConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    original_config = _export_config_to_huggingface(config)

    safetensors_weights_manager = SafeTensorsWeightsManager(pretrained_model_name_or_path)
    state_dict = _export_state_dict_to_huggingface(safetensors_weights_manager, config.num_layers)

    SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)
    original_config.save_pretrained(save_path)

    original_generation_config = GenerationConfig.from_model_config(original_config)
    original_generation_config.save_pretrained(save_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        tokenizer.save_pretrained(save_path, legacy_format=False)
    except:
        pass


def _export_config_to_huggingface(config: GPTBaseConfig) -> GPTBigCodeConfig:
    assert config.normalization_function == "layernorm"
    assert config.position_embedding_type == "learned_absolute"
    assert config.m_emb is None
    assert config.m_residual is None
    assert config.m_width is None

    num_attention_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads")
    num_key_value_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_key_value_heads")
    attention_head_type = get_attention_head_type(num_attention_heads, num_key_value_heads)

    assert attention_head_type in ["mha", "mqa"]
    assert config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "attention_multiplier") is None

    original_config = GPTBigCodeConfig(
        vocab_size=config.vocab_size,
        n_positions=config.max_position_embeddings,
        n_embd=config.hidden_size,
        n_layer=config.num_layers,
        n_head=num_attention_heads,
        n_inner=config.check_equal_for_all_and_get_value("mlp_blocks", "intermediate_size"),
        activation_function=config.check_equal_for_all_and_get_value(
            "mlp_blocks", "activation_function", "gelu_pytorch_tanh"
        ),
        embedding_dropout=config.embedding_dropout,
        attn_pdrop=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "softmax_dropout"),
        layer_norm_epsilon=config.layer_norm_epsilon,
        initializer_range=config.initializer_range,
        use_cache=config.use_cache,
        multi_query=attention_head_type == "mqa",
        tie_word_embeddings=config.tie_word_embeddings,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        architectures=[GPTBigCodeForCausalLM.__name__],
    )

    return original_config


def _export_state_dict_to_huggingface(safetensors_weights_manager: SafeTensorsWeightsManager, num_layers: int) -> dict:
    state_dict = {key: safetensors_weights_manager.get_tensor(key) for key in safetensors_weights_manager}

    for layer_idx in range(num_layers):
        # sequence_mixer.c_attn
        state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.sequence_mixer.c_attn.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.sequence_mixer.c_attn.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.attn.c_attn.bias"] = bias

        # sequence_mixer.c_proj
        state_dict[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.sequence_mixer.c_proj.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.attn.c_proj.bias"] = bias

        # mlp_block.c_fc
        state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.mlp_block.c_fc.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.mlp_block.c_fc.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.bias"] = bias

        # mlp_block.c_proj
        state_dict[f"transformer.h.{layer_idx}.mlp.c_proj.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"
        )

        bias = state_dict.pop(f"transformer.h.{layer_idx}.mlp_block.c_proj.bias", None)
        if bias is not None:
            state_dict[f"transformer.h.{layer_idx}.mlp.c_proj.bias"] = bias

    return state_dict
