# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import GraniteConfig, GraniteForCausalLM

from ..models import GPTBaseConfig


def _import_granite_config(original_config: GraniteConfig, **kwargs) -> GPTBaseConfig:
    assert original_config.hidden_act == "silu"
    assert original_config.mlp_bias == original_config.attention_bias
    use_interleaved_weights = kwargs.pop("use_interleaved_weights", False)

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
        m_emb=None if original_config.embedding_multiplier == 1 else original_config.embedding_multiplier,
        m_residual=None if original_config.residual_multiplier == 1 else original_config.residual_multiplier,
        m_width=None if original_config.logits_scaling == 1 else original_config.logits_scaling,
        sequence_mixer_blocks=[
            {
                "sequence_mixer_type": "softmax_attention",
                "add_bias": original_config.attention_bias,
                "num_attention_heads": original_config.num_attention_heads,
                "num_key_value_heads": original_config.num_key_value_heads,
                "attention_multiplier": original_config.attention_multiplier,
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
                "use_interleaved_weights": use_interleaved_weights,
            }
            for _ in range(original_config.num_hidden_layers)
        ],
    )

    assert len(kwargs) == 0

    return config


def _export_granite_config(config: GPTBaseConfig) -> GraniteConfig:
    assert config.normalization_function == "rmsnorm"
    assert config.position_embedding_type == "rope"

    config.check_equal_for_all_and_get_value("mlp_blocks", "activation_function", "swiglu")
    config.check_equal_for_all_and_get_value("mlp_blocks", "mlp_type", "MLP")

    original_config = GraniteConfig(
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
        embedding_multiplier=1 if config.m_emb is None else config.m_emb,
        residual_multiplier=1 if config.m_residual is None else config.m_residual,
        logits_scaling=1 if config.m_width is None else config.m_width,
        attention_multiplier=config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "attention_multiplier"),
        architectures=[GraniteForCausalLM.__name__],
    )

    return original_config
