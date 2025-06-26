# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from .config import CommonConfig
from .models import (
    GPTBaseConfig,
    LadderResidualConfig,
    fix_gpt_base_unsharded_state_dict,
    unshard_gpt_base_tensor_parallel_state_dicts,
)


_UNSHARD_STATE_DICT_FUNCTIONS = {
    GPTBaseConfig.model_type: unshard_gpt_base_tensor_parallel_state_dicts,
    LadderResidualConfig.model_type: unshard_gpt_base_tensor_parallel_state_dicts,
}


def unshard_tensor_parallel_state_dicts(
    config: GPTBaseConfig,
    tensor_parallel_state_dicts: list[dict],
    prefix: str = "",
    check_correctness: bool = True,
) -> dict:
    if config.model_type in _UNSHARD_STATE_DICT_FUNCTIONS:
        return _UNSHARD_STATE_DICT_FUNCTIONS[config.model_type](
            config=config,
            tensor_parallel_state_dicts=tensor_parallel_state_dicts,
            prefix=prefix,
            check_correctness=check_correctness,
        )

    raise ValueError(f"unsupported `model_type` ({config.model_type})")


_FIX_UNSHARDED_STATE_DICT_FUNCTIONS = {
    GPTBaseConfig.model_type: fix_gpt_base_unsharded_state_dict,
    LadderResidualConfig.model_type: fix_gpt_base_unsharded_state_dict,
}


def fix_unsharded_state_dict(
    config: CommonConfig, state_dict: dict, tensor_parallel_world_size: int, prefix: str = ""
) -> dict:
    if config.model_type in _FIX_UNSHARDED_STATE_DICT_FUNCTIONS:
        return _FIX_UNSHARDED_STATE_DICT_FUNCTIONS[config.model_type](
            config=config, state_dict=state_dict, tensor_parallel_world_size=tensor_parallel_world_size, prefix=prefix
        )

    raise ValueError(f"unsupported `model_type` ({config.model_type})")
