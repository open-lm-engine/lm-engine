# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.distributed

from ......parallel import ProcessGroupManager


class RingComm:
    def __init__(self) -> RingComm:
        self._ops = []
        self._reqs = None

        self.process_group = ProcessGroupManager.get_context_parallel_group()
        self.rank = ProcessGroupManager.get_context_parallel_rank()
        self.world_size = ProcessGroupManager.get_context_parallel_world_size()
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

    def send_recv(self, to_send: torch.Tensor, recv_tensor: torch.Tensor | None = None) -> torch.Tensor:
        res = torch.empty_like(to_send) if recv_tensor is None else recv_tensor

        send_op = torch.distributed.P2POp(torch.distributed.isend, to_send, self.send_rank, group=self.process_group)
        recv_op = torch.distributed.P2POp(torch.distributed.irecv, res, self.recv_rank, group=self.process_group)

        self._ops.append(send_op)
        self._ops.append(recv_op)

        return res

    def commit(self) -> None:
        if self._reqs is not None:
            raise RuntimeError("commit called twice")

        self._reqs = torch.distributed.batch_isend_irecv(self._ops)

    def wait(self) -> None:
        if self._reqs is None:
            raise RuntimeError("wait called before commit")

        for req in self._reqs:
            req.wait()

        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: torch.Tensor | None = None,
        v_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_k = self.send_recv(k, k_buffer)
        next_v = self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v


class _AllToAllRotater(_RingRotater):
    """Use all_to_all to send the kv to the next rank."""

    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._buffer: torch.Tensor | None = None

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        curr_buffer = curr_buffer.contiguous()
        size = dist.get_world_size(self._pg)
        dsts = list(range(1, size)) + [0]
        self._buffer = ft_c.permute_tensor(curr_buffer, dsts, self._pg)

    def next_buffer(self) -> torch.Tensor:
        assert self._buffer is not None
        return _maybe_wait(self._buffer)


class _AllGatherRotater(_RingRotater):
    """
    Allgather the kv and return only the required kv.
    Only one communication will be done.
    """

    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._aggregated_buffer: torch.Tensor | None = None
        self._idx = 0

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        # We only need to perform allgather once.
        self._idx += 1
        if self._aggregated_buffer is None:
            self._aggregated_buffer = ft_c.all_gather_tensor(curr_buffer.contiguous(), gather_dim=0, group=self._pg)

    def next_buffer(self) -> torch.Tensor:
        rank = dist.get_rank(self._pg)
        idx = rank - self._idx

        assert self._aggregated_buffer is not None
        self._aggregated_buffer = _maybe_wait(self._aggregated_buffer)
        return self._aggregated_buffer.chunk(dist.get_world_size(self._pg))[idx]


def _create_rotater(pg: dist.ProcessGroup, seq_dim: int, method: _RotateMethod | None = None) -> _RingRotater:
    if method is None:
        method = _cp_options.rotate_method

    if method == _RotateMethod.ALL_TO_ALL:
        return _AllToAllRotater(pg, seq_dim)
    elif method == _RotateMethod.ALL_GATHER:
        return _AllGatherRotater(pg, seq_dim)
    else:
        raise NotImplementedError(f"Unknown method {method}")
