# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.distributed

from ...parallel import ProcessGroupManager


class _RingShiftFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, src_global_rank: int, dst_global_rank: int, group) -> torch.Tensor:
        ctx.src_global_rank = src_global_rank
        ctx.dst_global_rank = dst_global_rank
        ctx.group = group

        recv_buf = torch.empty_like(x)
        ops = [
            torch.distributed.P2POp(torch.distributed.isend, x, dst_global_rank, group),
            torch.distributed.P2POp(torch.distributed.irecv, recv_buf, src_global_rank, group),
        ]
        for req in torch.distributed.batch_isend_irecv(ops):
            req.wait()
        return recv_buf

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple:
        recv_buf = torch.empty_like(grad)
        ops = [
            torch.distributed.P2POp(torch.distributed.isend, grad.contiguous(), ctx.src_global_rank, ctx.group),
            torch.distributed.P2POp(torch.distributed.irecv, recv_buf, ctx.dst_global_rank, ctx.group),
        ]
        for req in torch.distributed.batch_isend_irecv(ops):
            req.wait()
        return recv_buf, None, None, None


class SendRecvRotater:
    """Ring-shift rotater using point-to-point send/recv.

    Replaces AllGatherRotater for the common shift=1 case: sends to rank+1,
    receives from rank-1, costing O(x) data instead of O(x * world_size).
    """

    _buffer: torch.Tensor | None = None
    _reqs: list | None = None

    def exchange_buffers(self, x: torch.Tensor, with_grad: bool, shift: int = 1) -> None:
        x = x.contiguous()
        world_size = ProcessGroupManager.get_context_parallel_world_size()
        rank = ProcessGroupManager.get_context_parallel_rank()
        group = ProcessGroupManager.get_context_parallel_group()

        global_ranks = torch.distributed.get_process_group_ranks(group)
        src_global_rank = global_ranks[(rank - shift) % world_size]
        dst_global_rank = global_ranks[(rank + shift) % world_size]

        if with_grad:
            self._buffer = _RingShiftFunction.apply(x, src_global_rank, dst_global_rank, group)
            self._reqs = None
        else:
            recv_buf = torch.empty_like(x)
            ops = [
                torch.distributed.P2POp(torch.distributed.isend, x, dst_global_rank, group),
                torch.distributed.P2POp(torch.distributed.irecv, recv_buf, src_global_rank, group),
            ]
            self._reqs = torch.distributed.batch_isend_irecv(ops)
            self._buffer = recv_buf

    def next_buffer(self) -> torch.Tensor:
        assert self._buffer is not None
        if self._reqs is not None:
            for req in self._reqs:
                req.wait()
            self._reqs = None
        x = self._buffer
        self._buffer = None
        return x
