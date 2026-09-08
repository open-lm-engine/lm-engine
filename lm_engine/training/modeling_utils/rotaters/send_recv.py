# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.distributed

from ...parallel import ProcessGroupManager


@torch.library.custom_op("lm_engine::_send_recv_op", mutates_args={"y"})
def _send_recv_op(x: torch.Tensor, y: torch.Tensor, shift: int) -> None:
    world_size = ProcessGroupManager.get_context_parallel_world_size()
    rank = ProcessGroupManager.get_context_parallel_rank()
    group = ProcessGroupManager.get_context_parallel_group()

    global_ranks = torch.distributed.get_process_group_ranks(group)
    src_global_rank = global_ranks[(rank - shift) % world_size]
    dst_global_rank = global_ranks[(rank + shift) % world_size]

    ops = [
        torch.distributed.P2POp(torch.distributed.isend, x, dst_global_rank, group),
        torch.distributed.P2POp(torch.distributed.irecv, y, src_global_rank, group),
    ]

    for req in torch.distributed.batch_isend_irecv(ops):
        req.wait()


class _SendRecv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, shift: int) -> torch.Tensor:
        ctx.shape = x.size()
        ctx.dtype = x.dtype
        ctx.shift = shift

        x = x.contiguous()
        y = torch.empty_like(x)

        _send_recv_op(x=x, y=y, shift=shift)

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple:
        dy = dy.contiguous()
        dx = torch.empty(ctx.shape, dtype=ctx.dtype, device=dy.device)

        _send_recv_op(x=dy, y=dx, shift=-ctx.shift)

        return dx, None


def send_recv(x: torch.Tensor, shift: int = 1) -> torch.Tensor:
    return _SendRecv.apply(x, shift)
