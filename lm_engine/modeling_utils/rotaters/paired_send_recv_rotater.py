# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.distributed

from ...parallel import ProcessGroupManager


def _get_neighbor_global_ranks(shift: int) -> tuple[int, int]:
    world_size = ProcessGroupManager.get_context_parallel_world_size()
    rank = ProcessGroupManager.get_context_parallel_rank()
    group = ProcessGroupManager.get_context_parallel_group()

    global_ranks = torch.distributed.get_process_group_ranks(group)
    src_global_rank = global_ranks[(rank - shift) % world_size]
    dst_global_rank = global_ranks[(rank + shift) % world_size]

    return src_global_rank, dst_global_rank


class _BridgeGrad(torch.autograd.Function):
    """Ties an already-received buffer to `x`'s autograd graph. The actual receive happens earlier, in
    `PairedSendRecvRotater.exchange_buffers`/`next_buffer`; forward here is just a pass-through so `x`
    is wired into the graph. Backward sends the incoming gradient back to `src_global_rank`, and
    receives (as `x`'s gradient) whatever `dst_global_rank` computed for its own use of `x`."""

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, recv_buf: torch.Tensor, src_global_rank: int, dst_global_rank: int, group
    ) -> torch.Tensor:
        ctx.src_global_rank = src_global_rank
        ctx.dst_global_rank = dst_global_rank
        ctx.group = group
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype

        return recv_buf

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple:
        grad = grad.contiguous()
        grad_x = torch.empty(ctx.x_shape, dtype=ctx.x_dtype, device=grad.device)
        ops = [
            torch.distributed.P2POp(torch.distributed.isend, grad, ctx.src_global_rank, ctx.group),
            torch.distributed.P2POp(torch.distributed.irecv, grad_x, ctx.dst_global_rank, ctx.group),
        ]
        for req in torch.distributed.batch_isend_irecv(ops):
            req.wait()

        return grad_x, None, None, None, None


class PairedSendRecvRotater:
    """Point-to-point rotater that sends and receives together, with the same `exchange_buffers` +
    `next_buffer` shape as `AllGatherRotater`, but allows the sent and received tensors to differ
    (each rank exchanges genuinely different data with its neighbors in the same round, rather than
    rotating a single buffer). `exchange_buffers` only issues the send/recv; `next_buffer` waits and
    returns the result, same split as `AllGatherRotater`.
    """

    _buffer: torch.Tensor | None = None
    _reqs: list | None = None
    _x: torch.Tensor | None = None
    _with_grad: bool = False
    _src_global_rank: int | None = None
    _dst_global_rank: int | None = None

    def exchange_buffers(
        self,
        x: torch.Tensor,
        with_grad: bool,
        recv_shape: tuple[int, ...],
        recv_dtype: torch.dtype,
        shift: int = 1,
    ) -> None:
        x = x.contiguous()
        src_global_rank, dst_global_rank = _get_neighbor_global_ranks(shift)
        group = ProcessGroupManager.get_context_parallel_group()

        recv_buf = torch.empty(recv_shape, dtype=recv_dtype, device=x.device)
        ops = [
            torch.distributed.P2POp(torch.distributed.isend, x, dst_global_rank, group),
            torch.distributed.P2POp(torch.distributed.irecv, recv_buf, src_global_rank, group),
        ]

        self._reqs = torch.distributed.batch_isend_irecv(ops)
        self._buffer = recv_buf
        self._x = x
        self._with_grad = with_grad
        self._src_global_rank = src_global_rank
        self._dst_global_rank = dst_global_rank

    def next_buffer(self) -> torch.Tensor:
        assert self._buffer is not None

        for req in self._reqs:
            req.wait()
        self._reqs = None

        x = self._buffer
        self._buffer = None

        if self._with_grad:
            group = ProcessGroupManager.get_context_parallel_group()
            x = _BridgeGrad.apply(self._x, x, self._src_global_rank, self._dst_global_rank, group)

        self._x = None

        return x
