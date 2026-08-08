# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.distributed

from ...parallel import ProcessGroupManager


# NOTE: `x` is annotated as mutated even though a send only reads it, so that torch compile keeps the
# collective rather than eliminating a call that has no outputs. The annotation bumps `x`'s version
# counter, so don't hand this a tensor that autograd has saved for backward.
@torch.library.custom_op("lm_engine::send", mutates_args={"x"})
def send(x: torch.Tensor, shift: int = 1) -> None:
    """Blocking send of `x` to the rank `shift` places ahead in the context parallel group."""

    world_size = ProcessGroupManager.get_context_parallel_world_size()
    rank = ProcessGroupManager.get_context_parallel_rank()
    group = ProcessGroupManager.get_context_parallel_group()

    global_ranks = torch.distributed.get_process_group_ranks(group)
    dst_global_rank = global_ranks[(rank + shift) % world_size]

    torch.distributed.send(x.contiguous(), dst_global_rank, group)
