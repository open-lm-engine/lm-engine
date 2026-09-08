# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ...custom_op import ctx_save_for_backward, xma_op


@triton.jit
def _pack_unpack_sequence_triton_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    S,
    N,
    PADDING_SIDE: tl.constexpr,
    PACK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    BLOCK_ID_S = tl.program_id(0)
    BLOCK_ID_B = tl.program_id(1)

    cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_ID_B * cu_seqlens_stride[0]
    start = tl.load(cu_seqlens_ptrs)
    end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0])
    seqlens = end - start

    pad_tokens = (S - seqlens) if PADDING_SIDE == "left" else 0

    if (PADDING_SIDE == "left" and BLOCK_ID_S >= pad_tokens) or (PADDING_SIDE == "right" and BLOCK_ID_S < seqlens):
        BLOCK = tl.arange(0, BLOCK_SIZE)
        NUM_BLOCKS = tl.cdiv(N, BLOCK_SIZE)

        if PACK:
            x_ptrs = x_ptr + BLOCK_ID_B * x_stride[0] + BLOCK_ID_S * x_stride[1] + BLOCK * x_stride[-1]
            y_ptrs = y_ptr + (start + BLOCK_ID_S - pad_tokens) * y_stride[0] + BLOCK * y_stride[-1]
        else:
            x_ptrs = x_ptr + (start + BLOCK_ID_S - pad_tokens) * x_stride[0] + BLOCK * x_stride[-1]
            y_ptrs = y_ptr + BLOCK_ID_B * y_stride[0] + BLOCK_ID_S * y_stride[1] + BLOCK * y_stride[-1]

        for _ in range(NUM_BLOCKS):
            MASK = BLOCK < N

            x = tl.load(x_ptrs, mask=MASK)
            x_ptrs += BLOCK_SIZE * x_stride[-1]

            tl.store(y_ptrs, x, mask=MASK)
            y_ptrs += BLOCK_SIZE * y_stride[-1]

            BLOCK += BLOCK_SIZE


@xma_op(mutates_args={"y"})
def _pack_unpack_sequence_triton(
    x: torch.Tensor, y: torch.Tensor, cu_seqlens: torch.Tensor, padding_side: str, pack: bool
) -> None:
    if pack:
        B, S = x.size()[:2]
        N = x.numel() // (B * S)
    else:
        B, S = y.size()[:2]
        N = y.numel() // (B * S)

    BLOCK_SIZE = 4096
    NUM_WARPS = 32

    _pack_unpack_sequence_triton_kernel[S, B](
        x_ptr=x,
        x_stride=x.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        cu_seqlens_ptr=cu_seqlens,
        cu_seqlens_stride=cu_seqlens.stride(),
        S=S,
        N=N,
        PADDING_SIDE=padding_side,
        PACK=pack,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )


class _PackSequenceTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
    ) -> torch.Tensor:
        ctx_save_for_backward(ctx, cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()

        y = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        _pack_unpack_sequence_triton(x=x, y=y, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=True)

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        cu_seqlens = ctx.saved_tensors[0]

        dx = torch.zeros(*ctx.x_shape, device=dy.device, dtype=dy.dtype)
        _pack_unpack_sequence_triton(x=dy, y=dx, cu_seqlens=cu_seqlens, padding_side=ctx.padding_side, pack=False)

        return dx, None, None, None


class _UnpackSequenceTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
    ) -> torch.Tensor:
        ctx_save_for_backward(ctx, cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()

        y = torch.zeros(*output_shape, device=x.device, dtype=x.dtype)
        _pack_unpack_sequence_triton(x=x, y=y, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=False)

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        cu_seqlens = ctx.saved_tensors[0]

        dx = torch.empty(ctx.x_shape, device=dy.device, dtype=dy.dtype)
        _pack_unpack_sequence_triton(x=dy, y=dx, cu_seqlens=cu_seqlens, padding_side=ctx.padding_side, pack=True)

        return dx, None, None, None
