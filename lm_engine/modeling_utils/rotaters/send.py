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


class _Send(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, shift: int) -> torch.Tensor:
        ctx.shape = x.size()
        ctx.dtype = x.dtype
        ctx.shift = shift

        x = x.contiguous()
        _send_op(x=x, shift=shift)

        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        dx = torch.empty(ctx.shape, dtype=ctx.dtype, device=grad_output.device)
        _recv_op(y=dx, shift=-ctx.shift)

        return dx, None


def send(x: torch.Tensor, shift: int = 1) -> torch.Tensor:
    return _Send.apply(x, shift)
