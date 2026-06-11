# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from .models import GPTBaseConfig, LadderResidualConfig, unshard_gpt_base_tensor_parallel_state_dicts


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
