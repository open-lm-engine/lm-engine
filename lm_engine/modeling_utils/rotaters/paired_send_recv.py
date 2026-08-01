# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.distributed

from ...parallel import ProcessGroupManager


@torch.library.custom_op("lm_engine::_send_op", mutates_args=())
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


class _Recv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dummy: torch.Tensor, shape: torch.Size, dtype: torch.dtype, shift: int) -> torch.Tensor:
        ctx.shift = shift

        y = torch.empty(shape, dtype=dtype, device=dummy.device)
        _recv_op(y=y, shift=shift)

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        grad_output = grad_output.contiguous()
        _send_op(x=grad_output, shift=-ctx.shift)

        return None, None, None, None


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


def send(x: torch.Tensor, shift: int = 1) -> torch.Tensor:
    """Sends `x` forward; backward receives (from the same peer) the gradient it should propagate."""
    return _Send.apply(x, shift)


def recv(shape: torch.Size, dtype: torch.dtype, device: torch.device, shift: int = 1) -> torch.Tensor:
    """Receives a tensor forward; backward sends the incoming gradient back to whoever sent it."""
    dummy = torch.empty(0, device=device, requires_grad=True)
    return _Recv.apply(dummy, shape, dtype, shift)


def send_recv(x: torch.Tensor, shift: int = 1) -> torch.Tensor:
    return _SendRecv.apply(x, shift)
