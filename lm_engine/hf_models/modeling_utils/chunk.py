# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


class _ContiguousChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, chunks: int, dim: int) -> torch.Tensor:
        ctx.dim = dim
        x = x.chunk(chunks, dim=dim)
        return tuple(i.contiguous() for i in x)

    @staticmethod
    def backward(ctx, *dy: tuple[torch.Tensor]) -> tuple[torch.Tensor, None, None]:
        return torch.cat(dy, dim=ctx.dim), None, None


def contiguous_chunk(x: torch.Tensor, chunks: int, dim: int = 0) -> tuple[torch.Tensor]:
    return _ContiguousChunk.apply(x, chunks, dim)
