# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor, all_gather_tensor

from ...parallel import ProcessGroupManager


class AllGatherRotater:
    _buffer: torch.Tensor | None = None

    def exchange_buffers(self, x: torch.Tensor, with_grad: bool) -> None:
        group = ProcessGroupManager.get_context_parallel_group()

        if not with_grad:
            x = x.detach()

        if with_grad:
            self._buffer = all_gather_tensor(x, gather_dim=0, group=group)
        else:
            with torch.no_grad():
                self._buffer = all_gather_tensor(x, gather_dim=0, group=group)

    def next_buffer(self) -> torch.Tensor:
        assert self._buffer is not None
        x = self._buffer
        self._buffer = None

        if isinstance(x, AsyncCollectiveTensor):
            x = x.wait()

        rank = ProcessGroupManager.get_context_parallel_rank()
        world_size = ProcessGroupManager.get_context_parallel_world_size()

        x = x.chunk(world_size)[(rank - 1) % world_size]

        return x
