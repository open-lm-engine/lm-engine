# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from ...utils import SafeTensorsWeightsManager, divide_if_divisible, download_repo
from ..modeling_utils import (
    get_attention_head_type,
    interleave_query_key_value_tensor_for_attention,
    split_query_key_value_tensor_for_attention,
)
from ..models import GPTBaseConfig


# TODO remove try except once GraniteMoeHybrid is added into HF
try:
    from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM
except:
    GraniteMoeHybridConfig = None
    GraniteMoeHybridForCausalLM = None


def import_from_huggingface_granitemoehybrid(pretrained_model_name_or_path: str, save_path: str) -> None:
    original_config, tokenizer, downloaded_model_path = download_repo(pretrained_model_name_or_path)
    config = _import_config_from_huggingface(original_config)
    num_attention_heads = config.check_equal_for_all_and_get_value(
        "sequence_mixer_blocks", "num_attention_heads", sequence_mixer_type="softmax_attention"
    )

    safetensors_weights_manager = SafeTensorsWeightsManager(downloaded_model_path)
    state_dict = _import_state_dict_from_huggingface(
        safetensors_weights_manager,
        config.num_layers,
        [block.sequence_mixer_type for block in config.sequence_mixer_blocks],
        num_attention_heads,
        config.check_equal_for_all_and_get_value(
            "sequence_mixer_blocks", "num_key_value_heads", sequence_mixer_type="softmax_attention"
        ),
        config.hidden_size // num_attention_heads,
    )

    SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)
    config.save_pretrained(save_path)

    generation_config = GenerationConfig.from_model_config(config)
    generation_config.save_pretrained(save_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path, legacy_format=False)


def _import_sequence_mixer_config(original_config: GraniteMoeHybridConfig) -> list[dict]:
    configs = []
    for layer_idx in range(original_config.num_hidden_layers):
        layer_type = original_config.layer_types[layer_idx]

        if layer_type == "attention":
            config = {
                "sequence_mixer_type": "softmax_attention",
                "num_attention_heads": original_config.num_attention_heads,
                "num_key_value_heads": original_config.num_key_value_heads,
                "attention_multiplier": original_config.attention_multiplier,
                "add_bias": original_config.attention_bias,
                "softmax_dropout": original_config.attention_dropout,
            }
        elif layer_type == "mamba":
            config = {
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

        configs.append(config)

    return configs


def _import_config_from_huggingface(original_config: GraniteMoeHybridConfig) -> GPTBaseConfig:
    assert original_config.hidden_act == "silu"
    assert not original_config.attention_bias

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
        sequence_mixer_blocks=_import_sequence_mixer_config(original_config),
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
    sequence_mixer_block_types: str,
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

        if sequence_mixer_block_types[layer_idx] == "mamba2":
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.conv1d.weight"] = (
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mamba.conv1d.weight")
            )
            if safetensors_weights_manager.has_tensor(f"model.layers.{layer_idx}.mamba.conv1d.bias"):
                state_dict[f"transformer.h.{layer_idx}.sequence_mixer.conv1d.bias"] = (
                    safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mamba.conv1d.bias")
                )
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.in_proj.weight"] = (
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mamba.in_proj.weight")
            )
            if safetensors_weights_manager.has_tensor(f"model.layers.{layer_idx}.mamba.in_proj.bias"):
                state_dict[f"transformer.h.{layer_idx}.sequence_mixer.in_proj.bias"] = (
                    safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mamba.in_proj.bias")
                )
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.dt_bias"] = safetensors_weights_manager.get_tensor(
                f"model.layers.{layer_idx}.mamba.dt_bias"
            )
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.A_log"] = safetensors_weights_manager.get_tensor(
                f"model.layers.{layer_idx}.mamba.A_log"
            )
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.D"] = safetensors_weights_manager.get_tensor(
                f"model.layers.{layer_idx}.mamba.D"
            )
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.out_proj.weight"] = (
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mamba.out_proj.weight")
            )
            if safetensors_weights_manager.has_tensor(f"model.layers.{layer_idx}.mamba.out_proj.bias"):
                state_dict[f"transformer.h.{layer_idx}.sequence_mixer.out_proj.bias"] = (
                    safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mamba.out_proj.bias")
                )
            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.norm.weight"] = (
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.mamba.norm.weight")
            )
        elif sequence_mixer_block_types[layer_idx] == "softmax_attention":
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

            state_dict[f"transformer.h.{layer_idx}.sequence_mixer.c_proj.weight"] = (
                safetensors_weights_manager.get_tensor(f"model.layers.{layer_idx}.self_attn.o_proj.weight")
            )
        else:
            raise ValueError(f"unexpected sequence_mixer_type ({sequence_mixer_block_types[layer_idx]})")

    return state_dict


def export_to_huggingface_granitemoehybrid(pretrained_model_name_or_path: str, save_path: str) -> None:
    config: GPTBaseConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    original_config = _export_config_to_huggingface(config)

    safetensors_weights_manager = SafeTensorsWeightsManager(pretrained_model_name_or_path)
    state_dict = _export_state_dict_to_huggingface(
        safetensors_weights_manager,
        config.num_layers,
        sequence_mixer_block_types=_get_sequence_mixer_block_types(config),
        num_heads=original_config.num_attention_heads,
        num_key_value_heads=original_config.num_key_value_heads,
        head_dim=original_config.hidden_size // original_config.num_attention_heads,
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


def _get_sequence_mixer_block_types(config: GPTBaseConfig) -> list:
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


def _export_config_to_huggingface(config: GPTBaseConfig) -> GraniteMoeHybridConfig:
    assert config.normalization_function == "rmsnorm"
    assert config.position_embedding_type == "nope"

    config.check_equal_for_all_and_get_value("mlp_blocks", "add_bias", False)
    config.check_equal_for_all_and_get_value("mlp_blocks", "activation_function", "swiglu")
    config.check_equal_for_all_and_get_value("mlp_blocks", "mlp_type", "MoE")
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
        num_local_experts=config.check_equal_for_all_and_get_value("mlp_blocks", "num_experts"),
        num_experts_per_tok=config.check_equal_for_all_and_get_value("mlp_blocks", "num_experts_per_tok"),
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


def _export_state_dict_to_huggingface(
    safetensors_weights_manager: SafeTensorsWeightsManager,
    num_layers: int,
    sequence_mixer_block_types: list,
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

        if sequence_mixer_block_types[layer_idx] == "mamba":
            state_dict[f"model.layers.{layer_idx}.mamba.conv1d.weight"] = safetensors_weights_manager.get_tensor(
                f"transformer.h.{layer_idx}.sequence_mixer.conv1d.weight"
            )
            if safetensors_weights_manager.has_tensor(f"transformer.h.{layer_idx}.sequence_mixer.conv1d.bias"):
                state_dict[f"model.layers.{layer_idx}.mamba.conv1d.bias"] = safetensors_weights_manager.get_tensor(
                    f"transformer.h.{layer_idx}.sequence_mixer.conv1d.bias"
                )
            state_dict[f"model.layers.{layer_idx}.mamba.in_proj.weight"] = safetensors_weights_manager.get_tensor(
                f"transformer.h.{layer_idx}.sequence_mixer.in_proj.weight"
            )
            if safetensors_weights_manager.has_tensor(f"transformer.h.{layer_idx}.sequence_mixer.in_proj.bias"):
                state_dict[f"model.layers.{layer_idx}.mamba.in_proj.bias"] = safetensors_weights_manager.get_tensor(
                    f"transformer.h.{layer_idx}.sequence_mixer.in_proj.bias"
                )
            state_dict[f"model.layers.{layer_idx}.mamba.dt_bias"] = safetensors_weights_manager.get_tensor(
                f"transformer.h.{layer_idx}.sequence_mixer.dt_bias"
            )
            state_dict[f"model.layers.{layer_idx}.mamba.A_log"] = safetensors_weights_manager.get_tensor(
                f"transformer.h.{layer_idx}.sequence_mixer.A_log"
            )
            state_dict[f"model.layers.{layer_idx}.mamba.D"] = safetensors_weights_manager.get_tensor(
                f"transformer.h.{layer_idx}.sequence_mixer.D"
            )
            state_dict[f"model.layers.{layer_idx}.mamba.out_proj.weight"] = safetensors_weights_manager.get_tensor(
                f"transformer.h.{layer_idx}.sequence_mixer.out_proj.weight"
            )
            if safetensors_weights_manager.has_tensor(f"transformer.h.{layer_idx}.sequence_mixer.out_proj.bias"):
                state_dict[f"model.layers.{layer_idx}.mamba.out_proj.bias"] = safetensors_weights_manager.get_tensor(
                    f"transformer.h.{layer_idx}.sequence_mixer.out_proj.bias"
                )
            state_dict[f"model.layers.{layer_idx}.mamba.norm.weight"] = safetensors_weights_manager.get_tensor(
                f"transformer.h.{layer_idx}.sequence_mixer.norm.weight"
            )
        elif sequence_mixer_block_types[layer_idx] == "attention":
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
        else:
            raise ValueError(f"unexpected sequence_mixer_type ({sequence_mixer_block_types[layer_idx]})")

    return state_dict


def _split_and_reorder_for_glu(weight: torch.Tensor, dim: int) -> torch.Tensor:
    x, y = weight.chunk(2, dim=dim)
    weight = torch.cat([y, x], dim=dim)
    return weight
