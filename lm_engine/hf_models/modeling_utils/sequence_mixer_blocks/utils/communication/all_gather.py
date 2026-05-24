# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

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
            self._aggregated_buffer = ft_c.all_gather_tensor(curr_buffer.contiguous(), gather_dim=0, group=self._pg)

    def next_buffer(self) -> torch.Tensor:
        world_size = ProcessGroupManager.get_context_parallel_world_size()
        rank = ProcessGroupManager.get_context_parallel_rank()
        idx = rank - self._idx

        assert self._aggregated_buffer is not None
        self._aggregated_buffer = _maybe_wait(self._aggregated_buffer)
        return self._aggregated_buffer.chunk(world_size)[idx]
