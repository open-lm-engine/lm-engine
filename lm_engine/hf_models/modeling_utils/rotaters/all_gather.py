# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor, all_gather_tensor
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from ....parallel import ProcessGroupManager


class AllGatherRotater:
    _buffer: torch.Tensor | None = None
    _shift: int = 1

    def exchange_buffers(self, x: torch.Tensor, with_grad: bool, shift: int = 1) -> None:
        x = x.contiguous()
        mesh = ProcessGroupManager.get_context_parallel_mesh()
        self._shift = shift

        if with_grad:
            x = DTensor.from_local(x, device_mesh=mesh, placements=[Shard(0)])
            x = x.redistribute(placements=[Replicate()])
            x = x.to_local(grad_placements=[Partial()])
        else:
            x = all_gather_tensor(x, gather_dim=0, group=mesh)

        self._buffer = x

    def next_buffer(self) -> torch.Tensor:
        assert self._buffer is not None
        x = self._buffer
        self._buffer = None

        if isinstance(x, AsyncCollectiveTensor):
            x = x.wait()

        rank = ProcessGroupManager.get_context_parallel_rank()
        world_size = ProcessGroupManager.get_context_parallel_world_size()

        x = x.chunk(world_size)[(rank - self._shift) % world_size]

        return x
