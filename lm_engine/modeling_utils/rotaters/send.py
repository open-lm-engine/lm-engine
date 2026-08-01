# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.distributed

from ...parallel import ProcessGroupManager


# NOTE: to preserve autograd functionality, we need to annotate this as mutating x
@torch.library.custom_op("lm_engine::_send_op", mutates_args={"x"})
def _send_op(x: torch.Tensor, shift: int) -> None:
    world_size = ProcessGroupManager.get_context_parallel_world_size()
    rank = ProcessGroupManager.get_context_parallel_rank()
    group = ProcessGroupManager.get_context_parallel_group()

    global_ranks = torch.distributed.get_process_group_ranks(group)
    dst_global_rank = global_ranks[(rank + shift) % world_size]

    torch.distributed.send(x, dst_global_rank, group)


@torch.library.custom_op("lm_engine::_recv_op", mutates_args={"y"})
def _recv_op(y: torch.Tensor, shift: int) -> None:
    world_size = ProcessGroupManager.get_context_parallel_world_size()
    rank = ProcessGroupManager.get_context_parallel_rank()
    group = ProcessGroupManager.get_context_parallel_group()

    global_ranks = torch.distributed.get_process_group_ranks(group)
    src_global_rank = global_ranks[(rank - shift) % world_size]

    torch.distributed.recv(y, src_global_rank, group)


def send(x: torch.Tensor, shift: int = 1) -> None:
    """Blocking send of `x` to rank `cp_rank + shift`. Not differentiable.

    `_send_op` is annotated as mutating its input so that torch compile keeps the collective
    alive, so we hand it a private copy: mutating `x` itself would bump the version counter of
    a tensor autograd may have saved for backward.
    """
    _send_op(x=x.detach().clone(), shift=shift)
