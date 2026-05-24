# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ..accelerator import Accelerator
from .manager import ProcessGroupManager


def broadcast_tensor_parallel_input(tokens: dict, shape: tuple[int]) -> torch.Tensor:
    if ProcessGroupManager.is_tensor_parallel_enabled():
        device = Accelerator.get_current_device()

        if ProcessGroupManager.is_tensor_parallel_first_rank():
            tokens = tokens.to(device)
        else:
            tokens = torch.empty(shape, dtype=torch.long, device=device)

        torch.distributed.broadcast(
            tokens,
            src=ProcessGroupManager.get_tensor_parallel_first_rank(),
            group=ProcessGroupManager.get_tensor_parallel_group(),
        )
    else:
        tokens = tokens.to(Accelerator.get_current_device())

    return tokens
