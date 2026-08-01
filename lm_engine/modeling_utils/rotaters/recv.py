# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.distributed

from ...parallel import ProcessGroupManager


@torch.library.custom_op("lm_engine::recv", mutates_args={"x"})
def recv(x: torch.Tensor, shift: int = 1) -> None:
    """Blocking receive from the rank `shift` places behind in the context parallel group.

    `x` is written in place, so it has to be contiguous.
    """

    world_size = ProcessGroupManager.get_context_parallel_world_size()
    rank = ProcessGroupManager.get_context_parallel_rank()
    group = ProcessGroupManager.get_context_parallel_group()

    global_ranks = torch.distributed.get_process_group_ranks(group)
    src_global_rank = global_ranks[(rank - shift) % world_size]

    torch.distributed.recv(x, src_global_rank, group)
