# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed._functional_collectives import (
    AsyncCollectiveTensor,
    _expand_group,
    all_to_all_single_autograd,
    permute_tensor,
)
from torch.distributed.distributed_c10d import _find_or_create_pg_by_ranks_and_tag

from ...parallel import ProcessGroupManager


class AllToAllRotater:
    """Use all_to_all to send the kv to the next rank."""

    def __init__(self) -> AllToAllRotater:
        self._buffer: torch.Tensor | None = None

    def exchange_buffers(self, curr_buffer: torch.Tensor, with_grad: bool) -> None:
        curr_buffer = curr_buffer.contiguous()
        src_dst = list(range(1, ProcessGroupManager.get_context_parallel_world_size())) + [0]
        group = ProcessGroupManager.get_context_parallel_group()

        if with_grad:
            t, rankset, group_size = _expand_group(group)
            local_pg = _find_or_create_pg_by_ranks_and_tag(t, rankset, group_size)

            output_split_sizes = [0] * group_size
            input_split_sizes = [0] * group_size
            for src, dst in enumerate(src_dst):
                if src == torch.distributed.get_rank(local_pg):
                    input_split_sizes[dst] = x.numel()
                if dst == torch.distributed.get_rank(local_pg):
                    output_split_sizes[src] = x.numel()

            x = all_to_all_single_autograd(
                x, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group
            )
        else:
            x = permute_tensor(x, src_dst=src_dst, group=group)

        self._buffer = x

    def next_buffer(self) -> torch.Tensor:
        assert self._buffer is not None

        if isinstance(self._buffer, AsyncCollectiveTensor):
            self._buffer = self._buffer.wait()

        return self._buffer
