# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ..utils import Accelerator, ProcessGroupManager


def broadcast_tensor_parallel_input(tokens: dict, shape: tuple[int]) -> torch.Tensor:
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

    return tokens
