# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from ...gpt_base import GPTBaseConfig


def fix_gpt_base_unsharded_state_dict(
    config: GPTBaseConfig, state_dict: dict, tensor_parallel_world_size: int, prefix: str = ""
) -> dict:
    state_dict[prefix + "transformer.wte.weight"] = state_dict[prefix + "transformer.wte.weight"][
        : config.vocab_size, :
    ]

    return state_dict
