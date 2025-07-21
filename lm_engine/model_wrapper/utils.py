# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ..utils import ProcessGroupManager


def broadcast_tensor_parallel_input(tokens: dict, shape: tuple[int]) -> torch.Tensor:
    if ProcessGroupManager.is_tensor_parallel_first_rank():
        tokens = tokens.to(torch.cuda.current_device())
    else:
        tokens = torch.empty(shape, dtype=torch.long, device=torch.cuda.current_device())

    torch.distributed.broadcast(
        tokens,
        src=ProcessGroupManager.get_tensor_parallel_first_rank(),
        group=ProcessGroupManager.get_tensor_parallel_group(),
    )

    return tokens
