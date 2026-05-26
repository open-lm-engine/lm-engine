# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn.functional as F


def _partial_update(
    original: torch.Tensor,
    new: torch.Tensor,
    dim: int,
    n_chunks: int,
    idx: int,
    add: bool,
) -> torch.Tensor:
    """
    This API partially updates a chunk of ``original`` tensor. The ``original``
    tensor will be first chunked along ``dim`` dimension, then the ``idx`` chunk
    will be updated with ``new``. If ``add`` is True, the chunk will be added
    with ``new``, otherwise the chunk will be replaced by ``new``.

    The result is a tensor that is the same size as ``original``.
    """
    chunks = list(original.chunk(n_chunks, dim=dim))
    assert chunks[idx].size() == new.size(), (original.size(), new.size(), idx)
    if add:
        chunks[idx] += new
    else:
        chunks[idx] = new
    return torch.cat(chunks, dim=dim)


class _Merger:
    def __init__(self, seq_dim: int) -> _Merger:
        self._seq_dim = seq_dim
        self._out: torch.Tensor | None = None
        self._lse: torch.Tensor | None = None
        self._out_dtype = torch.float32
        self._lse_dtype = torch.float32

    def _merge_one(self, block_out: torch.Tensor, block_lse: torch.Tensor, partial: bool) -> None:
        block_lse = block_lse.transpose(1, 2)[..., None]
        assert block_lse.dim() == block_out.dim()

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
        self._merge_one(out.float(), lse.float(), partial)

    def results(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._out is not None
        assert self._lse is not None

        out = self._out.to(self._out_dtype)
        # Internally lse is kept as [B, S, H, 1] so it broadcasts against
        # out [B, S, H, D]. flash-attn's backward expects [B, H, S].
        lse = self._lse.squeeze(-1).transpose(1, 2).contiguous().to(self._lse_dtype)

        return out, lse
