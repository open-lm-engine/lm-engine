# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....modeling_utils import get_attention_head_type, is_glu
from ...gpt_base import GPTBaseConfig


def fix_gpt_base_unsharded_state_dict(
    config: GPTBaseConfig, state_dict: dict, tensor_parallel_world_size: int, prefix: str = ""
) -> dict:
    state_dict[prefix + "transformer.wte.weight"] = state_dict[prefix + "transformer.wte.weight"][
        : config.vocab_size, :
    ]

    for layer_idx in range(config.num_layers):
        block = config.sequence_mixer_blocks[layer_idx]

        if get_attention_head_type(block.num_attention_heads, block.num_key_value_heads) == "mqa":
            q_attn_w = state_dict.pop(f"{prefix}transformer.h.{layer_idx}.sequence_mixer.c_attn.q_attn.weight")
            kv_attn_w = state_dict.pop(f"{prefix}transformer.h.{layer_idx}.sequence_mixer.c_attn.kv_attn.weight")
            state_dict[f"{prefix}transformer.h.{layer_idx}.sequence_mixer.c_attn.weight"] = torch.cat(
                [q_attn_w, kv_attn_w]
            )

            if block.add_bias:
                q_attn_w = state_dict.pop(f"{prefix}transformer.h.{layer_idx}.sequence_mixer.c_attn.q_attn.bias")
                kv_attn_w = state_dict.pop(f"{prefix}transformer.h.{layer_idx}.sequence_mixer.c_attn.kv_attn.bias")
                state_dict[f"{prefix}transformer.h.{layer_idx}.sequence_mixer.c_attn.bias"] = torch.cat(
                    [q_attn_w, kv_attn_w]
                )

        block = config.mlp_blocks[layer_idx]

        if is_glu(block.activation_function):
            if block.mlp_type == "MLP":
                key = f"{prefix}transformer.h.{layer_idx}.mlp_block.c_fc.weight"
                weight = state_dict[key].chunk(tensor_parallel_world_size)
                weight = [w.chunk(2) for w in weight]
                w0 = torch.cat([w[0] for w in weight])
                w1 = torch.cat([w[1] for w in weight])
                state_dict[key] = torch.cat([w0, w1])

                if block.add_bias:
                    key = f"{prefix}transformer.h.{layer_idx}.mlp_block.c_fc.bias"
                    weight = state_dict[key].chunk(tensor_parallel_world_size)
                    weight = [w.chunk(2) for w in weight]
                    w0 = torch.cat([w[0] for w in weight])
                    w1 = torch.cat([w[1] for w in weight])
                    state_dict[key] = torch.cat([w0, w1])
            elif block.mlp_type == "MoE":
                assert not block.add_bias

                key = f"{prefix}transformer.h.{layer_idx}.mlp_block.c_fc.weight"
                weight = state_dict[key]
                weight = weight.chunk(tensor_parallel_world_size, dim=1)
                weight = [w.chunk(2, dim=1) for w in weight]
                w0 = torch.cat([w[0] for w in weight], dim=1)
                w1 = torch.cat([w[1] for w in weight], dim=1)
                state_dict[key] = torch.cat([w0, w1], dim=1)
            else:
                raise ValueError(f"unexpected mlp_type ({block.mlp_type})")

    return state_dict
