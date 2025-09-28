# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from ..models import GPTBaseConfig


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
