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


class _Merger:
    def __init__(self, seq_dim: int) -> _Merger:
        self._seq_dim = seq_dim
        self._out: torch.Tensor | None = None
        self._lse: torch.Tensor | None = None
        self._should_lse_squeeze = False
        self._out_dtype = torch.float32
        self._lse_dtype = torch.float32

    def _merge_one(self, block_out: torch.Tensor, block_lse: torch.Tensor, partial: bool) -> None:
        # The cuDNN backend preserves the last dimension for LSE.
        # Apply unsqueeze only if the input does not already have
        # the required dimensionality.
        if len(block_lse.shape) < len(block_out.shape):
            block_lse = block_lse.unsqueeze(dim=-1)
            self._should_lse_squeeze = True
        assert len(block_lse.shape) == len(block_out.shape)

        if self._lse is None:
            self._lse = block_lse
            self._out = block_out
        else:
            ROUND_ROBIN_CYCLE = 2
            assert self._lse is not None
            assert self._out is not None
            lse = self._lse.chunk(ROUND_ROBIN_CYCLE, dim=self._seq_dim)[1] if partial else self._lse
            out = self._out.chunk(ROUND_ROBIN_CYCLE, dim=self._seq_dim)[1] if partial else self._out

            # The algorithm from
            # github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
            # gives a relatively stable result.
            out = out - F.sigmoid(block_lse - lse) * (out - block_out)
            lse = lse - F.logsigmoid(lse - block_lse)
            if partial:
                self._lse = _partial_update(
                    self._lse,
                    lse,
                    dim=self._seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=1,
                    add=False,
                )
                self._out = _partial_update(
                    self._out,
                    out,
                    dim=self._seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=1,
                    add=False,
                )
            else:
                self._lse = lse
                self._out = out

    def step(self, out: torch.Tensor, lse: torch.Tensor, partial: bool) -> None:
        self._out_dtype = out.dtype
        self._lse_dtype = lse.dtype

        out = out.to(torch.float32)
        lse = lse.to(torch.float32)

        self._merge_one(out, lse, partial)

    def results(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._out is not None
        assert self._lse is not None
        out = self._out.to(self._out_dtype)
        if self._should_lse_squeeze:
            lse = self._lse.squeeze(-1).to(self._lse_dtype)
        else:
            lse = self._lse.to(self._lse_dtype)
        return out, lse
