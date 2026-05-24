# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.distributed

from .....parallel import ProcessGroupManager


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
