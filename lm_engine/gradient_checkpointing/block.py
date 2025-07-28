# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from ..utils import get_module_class_from_name


def block_checkpointing(
    model: nn.Module, block_name: str, num_blocks: int | None = None, use_reentrant: bool = False
) -> None:
    block_class = get_module_class_from_name(model, block_name)
    block_idx = 0

    def _whether_to_checkpoint(submodule: nn.Module) -> bool:
        nonlocal block_idx

        if isinstance(submodule, block_class):
            if num_blocks is None:
                return True

            block_idx += 1
            if block_idx <= num_blocks:
                return True
        return False

    checkpoint_wrapper_function = checkpoint_wrapper
    if use_reentrant:
        checkpoint_wrapper_function = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT)

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper_function, check_fn=_whether_to_checkpoint
    )
