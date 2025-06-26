# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from transformers import AutoModelForCausalLM

from ...modeling_utils import get_attention_head_type, split_query_key_value_tensor_for_attention
from ..gpt_base import GPTBaseConfig, GPTBaseForCausalLM
from .config import GPTCrossLayerConfig
from .main import GPTCrossLayerForCausalLM


def convert_gpt_base_to_gpt_crosslayer(
    original_config: GPTBaseConfig,
    original_model: GPTBaseForCausalLM,
    sharing_pattern: list[int] | None = None,
    **kwargs,
) -> GPTCrossLayerForCausalLM:
    config = GPTCrossLayerConfig(
        vocab_size=original_config.vocab_size,
        max_position_embeddings=original_config.max_position_embeddings,
        hidden_size=original_config.hidden_size,
        num_layers=original_config.num_layers,
        embedding_dropout=original_config.embedding_dropout,
        normalization_function=original_config.normalization_function,
        layer_norm_epsilon=original_config.layer_norm_epsilon,
        position_embedding_type=original_config.position_embedding_type,
        rope_theta=original_config.rope_theta,
        use_cache=original_config.use_cache,
        sharing_pattern=sharing_pattern,
        tie_word_embeddings=original_config.tie_word_embeddings,
        sequence_mixer_blocks=[i.to_dict() for i in original_config.sequence_mixer_blocks],
        mlp_blocks=[i.to_dict() for i in original_config.mlp_blocks],
    )
    model = AutoModelForCausalLM.from_config(config, torch_dtype=original_model.dtype, **kwargs)

    hidden_size = config.hidden_size
    num_attention_heads = config.check_equal_for_all_and_get_value("sequence_mixer_blocks", "num_attention_heads")
    head_dim = hidden_size // num_attention_heads

    state_dict = original_model.state_dict()

    sharing_pattern = config.sharing_pattern

    new_state_dict = {"transformer.wte.weight": state_dict["transformer.wte.weight"]}

    if "transformer.wpe.weight" in state_dict:
        new_state_dict["transformer.wpe.weight"] = state_dict["transformer.wpe.weight"]

    if "lm_head.weight" in state_dict:
        new_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]

    new_state_dict["transformer.ln_f.weight"] = state_dict["transformer.ln_f.weight"]
    if config.normalization_function == "layernorm":
        new_state_dict["transformer.ln_f.bias"] = state_dict["transformer.ln_f.bias"]

    for layer_idx in range(original_config.num_layers):
        num_key_value_heads = config.sequence_mixer_blocks[layer_idx].num_key_value_heads
        attention_head_type = get_attention_head_type(num_attention_heads, num_key_value_heads)
        add_bias = config.sequence_mixer_blocks[layer_idx].add_bias

        q_attn_weight, k_attn_weight, v_attn_weight = split_query_key_value_tensor_for_attention(
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_attn.weight"],
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attention_head_type,
        )
        if add_bias:
            q_attn_bias, k_attn_bias, v_attn_bias = split_query_key_value_tensor_for_attention(
                state_dict.pop(f"transformer.h.{layer_idx}.sequence_mixer.c_attn.bias"),
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                attention_head_type,
            )

        new_state_dict[f"transformer.h.{layer_idx}.sequence_mixer.q_attn.weight"] = q_attn_weight
        if add_bias:
            new_state_dict[f"transformer.h.{layer_idx}.sequence_mixer.q_attn.bias"] = q_attn_bias

        if layer_idx in sharing_pattern:
            new_state_dict[f"transformer.h.{layer_idx}.kv_proj.kv_attn.weight"] = torch.cat(
                [
                    k_attn_weight.view(num_key_value_heads, -1, hidden_size),
                    v_attn_weight.view(num_key_value_heads, -1, hidden_size),
                ],
                dim=1,
            ).view(-1, hidden_size)
            if add_bias:
                new_state_dict[f"transformer.h.{layer_idx}.kv_proj.kv_attn.bias"] = torch.cat(
                    [k_attn_bias.view(num_key_value_heads, -1), v_attn_bias.view(num_key_value_heads, -1)], dim=1
                ).view(-1)

            new_state_dict[f"transformer.h.{layer_idx}.kv_proj.ln.weight"] = state_dict[
                f"transformer.h.{layer_idx}.ln_1.weight"
            ]
            if config.normalization_function == "layernorm":
                new_state_dict[f"transformer.h.{layer_idx}.kv_proj.ln.bias"] = state_dict[
                    f"transformer.h.{layer_idx}.ln_1.bias"
                ]

        for ln in ["ln_1", "ln_2"]:
            new_state_dict[f"transformer.h.{layer_idx}.{ln}.weight"] = state_dict.pop(
                f"transformer.h.{layer_idx}.{ln}.weight"
            )
            if config.normalization_function == "layernorm":
                new_state_dict[f"transformer.h.{layer_idx}.{ln}.bias"] = state_dict.pop(
                    f"transformer.h.{layer_idx}.{ln}.bias"
                )

        add_bias = config.mlp_blocks[layer_idx].add_bias

        new_state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"
        )
        if add_bias:
            new_state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_proj.bias"] = state_dict.pop(
                f"transformer.h.{layer_idx}.sequence_mixer.c_proj.bias"
            )

        new_state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.mlp_block.c_fc.weight"
        )
        if add_bias:
            new_state_dict[f"transformer.h.{layer_idx}.mlp_block.c_fc.bias"] = state_dict.pop(
                f"transformer.h.{layer_idx}.mlp_block.c_fc.bias"
            )

        new_state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"] = state_dict.pop(
            f"transformer.h.{layer_idx}.mlp_block.c_proj.weight"
        )
        if add_bias:
            new_state_dict[f"transformer.h.{layer_idx}.mlp_block.c_proj.bias"] = state_dict.pop(
                f"transformer.h.{layer_idx}.mlp_block.c_proj.bias"
            )

    model.load_state_dict(new_state_dict)

    return config, model
