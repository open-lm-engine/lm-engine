# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


class _ContiguousChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, chunks: int, dim: int) -> tuple[torch.Tensor, ...]:
        ctx.dim = dim
        x = x.chunk(chunks, dim=dim)
        return tuple(i.contiguous() for i in x)

    @staticmethod
    def backward(ctx, *dy: tuple[torch.Tensor]) -> tuple[torch.Tensor, None, None]:
        dy = tuple(i.contiguous() for i in dy)
        return torch.cat(dy, dim=ctx.dim), None, None


def contiguous_chunk(x: torch.Tensor, chunks: int, dim: int = 0) -> tuple[torch.Tensor]:
    return _ContiguousChunk.apply(x, chunks, dim)


class _ContiguousSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, split_size: int | tuple[int, ...], dim: int) -> tuple[torch.Tensor, ...]:
        ctx.dim = dim
        x = x.split(split_size, dim=dim)
        return tuple(i.contiguous() for i in x)

    @staticmethod
    def backward(ctx, *dy: tuple[torch.Tensor]) -> tuple[torch.Tensor, None, None]:
        dy = tuple(i.contiguous() for i in dy)
        return torch.cat(dy, dim=ctx.dim), None, None


def contiguous_split(x: torch.Tensor, split_size: tuple[int, ...], dim: int = 0) -> tuple[torch.Tensor]:
    return _ContiguousSplit.apply(x, split_size, dim)
