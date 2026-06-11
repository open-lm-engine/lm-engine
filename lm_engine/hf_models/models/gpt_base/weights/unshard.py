# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from tqdm import trange

from ...gpt_base import GPTBaseConfig


def unshard_gpt_base_tensor_parallel_state_dicts(
    config: GPTBaseConfig,
    tensor_parallel_state_dicts: list[dict],
    prefix: str = "",
    check_correctness: bool = True,
) -> dict:
    # word embeddings
    output_state_dict = _get_embeddings_or_lm_head(
        tensor_parallel_state_dicts, prefix=prefix + "transformer.wte.weight", vocab_size=config.vocab_size
    )

    # positional embeddings if using learned positional embeddings
    if config.position_embedding_type == "learned_absolute":
        output_state_dict.update(
            _get_embeddings_or_lm_head(
                tensor_parallel_state_dicts,
                prefix=prefix + "transformer.wpe.weight",
                vocab_size=config.max_position_embeddings,
            )
        )

    # layers
    for layer_idx in trange(config.num_layers):
        block = config.sequence_mixer_blocks[layer_idx]

        # first layernorm
        output_state_dict.update(
            _get_layernorm(
                tensor_parallel_state_dicts,
                prefix=prefix + f"transformer.h.{layer_idx}.ln_1.",
                normalization_function=config.normalization_function,
                check_correctness=check_correctness,
            )
        )

        # attention
        output_state_dict.update(
            _get_attention(
                tensor_parallel_state_dicts,
                add_bias=block.add_bias,
                prefix=prefix + f"transformer.h.{layer_idx}.sequence_mixer.",
                check_correctness=check_correctness,
            )
        )

        # second layernorm
        output_state_dict.update(
            _get_layernorm(
                tensor_parallel_state_dicts,
                prefix=prefix + f"transformer.h.{layer_idx}.ln_2.",
                normalization_function=config.normalization_function,
                check_correctness=check_correctness,
            )
        )

        block = config.mlp_blocks[layer_idx]
        mlp_type = block.mlp_type

        # mlp
        if mlp_type == "MLP":
            output_state_dict.update(
                _get_mlp(
                    tensor_parallel_state_dicts,
                    add_bias=block.add_bias,
                    prefix=prefix + f"transformer.h.{layer_idx}.mlp_block.",
                    check_correctness=check_correctness,
                )
            )
        elif mlp_type == "MoE":
            output_state_dict.update(
                _get_moe(
                    tensor_parallel_state_dicts,
                    add_bias=block.add_bias,
                    prefix=prefix + f"transformer.h.{layer_idx}.mlp_block.",
                )
            )

    # final layernorm
    output_state_dict.update(
        _get_layernorm(
            tensor_parallel_state_dicts,
            prefix=prefix + f"transformer.ln_f.",
            normalization_function=config.normalization_function,
            check_correctness=check_correctness,
        )
    )

    if not config.tie_word_embeddings:
        output_state_dict.update(
            _get_embeddings_or_lm_head(
                tensor_parallel_state_dicts,
                prefix=prefix + "lm_head.weight",
                vocab_size=config.vocab_size,
                check_correctness=check_correctness,
            )
        )

    return output_state_dict


def _get_embeddings_or_lm_head(tensor_parallel_state_dicts: list[dict], prefix: str, vocab_size: int) -> dict:
    output = _concatenate_tensors_from_state_dicts(tensor_parallel_state_dicts, key=prefix, dim=0)

    assert output.shape[0] >= vocab_size
    output = output[:vocab_size, :]

    return {prefix: output}


def _get_layernorm(
    tensor_parallel_state_dicts: list[dict], prefix: str, normalization_function: str, check_correctness: bool
) -> dict:
    output = {
        prefix
        + "weight": _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "weight", check_correctness=check_correctness
        )
    }
    if normalization_function == "layernorm":
        output[prefix + "bias"] = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "bias", check_correctness=check_correctness
        )
    return output


def _get_attention(
    tensor_parallel_state_dicts: list[dict], add_bias: bool, prefix: str, check_correctness: bool
) -> dict:
    output = {
        prefix
        + "c_proj.weight": _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_proj.weight", dim=1
        )
    }
    if add_bias:
        output[prefix + "c_proj.bias"] = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "c_proj.bias", check_correctness=check_correctness
        )

    output[prefix + "c_attn.weight"] = _concatenate_tensors_from_state_dicts(
        tensor_parallel_state_dicts, key=prefix + "c_attn.weight", dim=0
    )
    if add_bias:
        output[prefix + "c_attn.bias"] = _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_attn.bias", dim=0
        )

    return output


def _get_mlp(tensor_parallel_state_dicts: list[dict], add_bias: bool, prefix: str, check_correctness: bool) -> dict:
    output = {
        prefix
        + "c_proj.weight": _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_proj.weight", dim=1
        )
    }
    if add_bias:
        output[prefix + "c_proj.bias"] = _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, key=prefix + "c_proj.bias", check_correctness=check_correctness
        )

    output[prefix + "c_fc.weight"] = _concatenate_tensors_from_state_dicts(
        tensor_parallel_state_dicts, key=prefix + "c_fc.weight", dim=0
    )
    if add_bias:
        output[prefix + "c_fc.bias"] = _concatenate_tensors_from_state_dicts(
            tensor_parallel_state_dicts, key=prefix + "c_fc.bias", dim=0
        )

    return output


def _concatenate_tensors_from_state_dicts(tensor_parallel_state_dicts: list[dict], key: str, dim: int) -> torch.Tensor:
    tensor_list = [state_dict[key] for state_dict in tensor_parallel_state_dicts]
    tensor = torch.cat(tensor_list, dim=dim)
    return tensor


def _get_once_from_state_dicts_with_check(
    tensor_parallel_state_dicts: list[dict], key: str, check_correctness: bool
) -> torch.Tensor:
    output: torch.Tensor = tensor_parallel_state_dicts[0][key]
    if check_correctness:
        for state_dict in tensor_parallel_state_dicts[1:]:
            assert output.equal(state_dict[key])
    return output


def _concatenate_tensors_from_moe(tensor_parallel_state_dicts: list[dict], key: str, dim: int) -> torch.Tensor:
    tensor_list = [state_dict[key] for state_dict in tensor_parallel_state_dicts]
    tensor = torch.cat(tensor_list, dim=dim)
    return tensor


def _get_moe(tensor_parallel_state_dicts: list[dict], add_bias: bool, prefix: str) -> dict:
    assert not add_bias

    output = {
        prefix
        + "gate.weight": _get_once_from_state_dicts_with_check(
            tensor_parallel_state_dicts, prefix + "gate.weight", True
        )
    }

    output[prefix + "c_fc.weight"] = _concatenate_tensors_from_state_dicts(
        tensor_parallel_state_dicts, prefix + "c_fc.weight", dim=1
    )

    output[prefix + "c_proj.weight"] = _concatenate_tensors_from_moe(
        tensor_parallel_state_dicts, prefix + "c_proj.weight", dim=2
    )
    return output
