# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor, all_gather_tensor

from ......parallel import ProcessGroupManager


class AllGatherRotater:
    """
    Allgather the kv and return only the required kv.
    Only one communication will be done.
    """

    def __init__(self, seq_dim: int) -> AllGatherRotater:
        self._seq_dim = seq_dim
        self._aggregated_buffer: torch.Tensor | None = None
        self._idx = 0

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        # We only need to perform allgather once.
        self._idx += 1
        if self._aggregated_buffer is None:
            self._aggregated_buffer = all_gather_tensor(curr_buffer.contiguous(), gather_dim=0, group=self._pg)

    def next_buffer(self) -> torch.Tensor:
        idx = ProcessGroupManager.get_context_parallel_rank() - self._idx

        assert self._aggregated_buffer is not None

        if isinstance(self._aggregated_buffer, AsyncCollectiveTensor):
            self._aggregated_buffer = self._aggregated_buffer.wait()

        return self._aggregated_buffer.chunk(ProcessGroupManager.get_context_parallel_world_size())[idx]
