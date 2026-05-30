# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor, permute_tensor

from .....parallel import ProcessGroupManager


class AllToAllRotater:
    """Use all_to_all to send the kv to the next rank."""

    def __init__(self, seq_dim: int) -> AllToAllRotater:
        self._seq_dim = seq_dim
        self._buffer: torch.Tensor | None = None

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        curr_buffer = curr_buffer.contiguous()
        dsts = list(range(1, ProcessGroupManager.get_context_parallel_world_size())) + [0]
        self._buffer = permute_tensor(curr_buffer, dsts, ProcessGroupManager.get_context_parallel_group())

    def next_buffer(self) -> torch.Tensor:
        assert self._buffer is not None

        if isinstance(self._buffer, AsyncCollectiveTensor):
            self._buffer = self._buffer.wait()

        return self._buffer
