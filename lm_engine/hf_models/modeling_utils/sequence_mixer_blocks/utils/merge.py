# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch


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
