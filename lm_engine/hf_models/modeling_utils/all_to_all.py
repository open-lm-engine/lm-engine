# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed._functional_collectives import RANK_TYPES, AsyncCollectiveTensor
from torch.distributed._functional_collectives import permute_tensor as _permute_tensor_no_grad

from ...parallel import ProcessGroupManager


class _PermuteTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, src_dst: list[int], group: RANK_TYPES) -> torch.Tensor:
        ctx.src_dst = src_dst
        x = _permute_tensor_no_grad(x, src_dst=src_dst, group=ProcessGroupManager.get_context_parallel_group())
        return x

    @staticmethod
    def backward(ctx, dx: torch.Tensor):
        src_dst = ctx.src_dst
        inv_src_dst = [0] * len(src_dst)
        for src, dst in enumerate(src_dst):
            inv_src_dst[dst] = src
        dx = _permute_tensor_no_grad(dx, src_dst=inv_src_dst, group=ProcessGroupManager.get_context_parallel_group())
        return dx, None, None


def permute_tensor(x: torch.Tensor, src_dst: list[int], group: RANK_TYPES, with_grad: bool) -> torch.Tensor:
    if with_grad:
        x = _PermuteTensor.apply(x, src_dst, group)
    else:
        x = _permute_tensor_no_grad(x, src_dst=src_dst, group=group)

    return x


class AllToAllRotater:
    """Use all_to_all to send the kv to the next rank."""

    def __init__(self, seq_dim: int) -> AllToAllRotater:
        self._seq_dim = seq_dim
        self._buffer: torch.Tensor | None = None

    def exchange_buffers(self, curr_buffer: torch.Tensor, with_grad: bool) -> None:
        curr_buffer = curr_buffer.contiguous()
        dsts = list(range(1, ProcessGroupManager.get_context_parallel_world_size())) + [0]
        self._buffer = permute_tensor(
            curr_buffer, src_dst=dsts, group=ProcessGroupManager.get_context_parallel_group(), with_grad=with_grad
        )

    def next_buffer(self) -> torch.Tensor:
        assert self._buffer is not None

        if isinstance(self._buffer, AsyncCollectiveTensor):
            self._buffer = self._buffer.wait()

        return self._buffer
