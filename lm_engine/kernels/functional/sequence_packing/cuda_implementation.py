# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Boolean, const_expr, range_constexpr

from ...custom_op import ctx_save_for_backward, xma_op
from ...cute_dsl_utils import get_fake_cute_tensor


class _PackUnpackSequenceCUDAKernel:
    def __init__(self, N: int, padding_side: str, pack: bool, BLOCK_SIZE: int) -> _PackUnpackSequenceCUDAKernel:
        self.N = N
        self.padding_side = padding_side
        self.pack = pack
        self.BLOCK_SIZE = BLOCK_SIZE

    @cute.jit
    def _copy_array(
        self,
        bXgX: cute.Tensor,
        bXgY: cute.Tensor,
        bXgC: cute.Tensor,
        copy_atom: cute.CopyAtom,
        shape: cute.Shape,
    ) -> None:
        vector_size = const_expr(128 // bXgX.element_type.width)
        N_vec = (self.N + vector_size - 1) // vector_size
        THREAD_ID = cute.arch.thread_idx()[0]

        for i in range(THREAD_ID, N_vec, self.BLOCK_SIZE):
            tXgX = cute.local_tile(bXgX, (vector_size,), (i,))
            tXgY = cute.local_tile(bXgY, (vector_size,), (i,))
            tXgC = cute.local_tile(bXgC, (vector_size,), (i,))

            rX = cute.make_rmem_tensor_like(tXgX)
            rC = cute.make_rmem_tensor_like(tXgC, dtype=Boolean)
            for j in range_constexpr(cute.size(rC)):
                rC[j] = cute.elem_less(tXgC[j], shape)

            is_within_boundary = cute.elem_less(tXgC[cute.size(tXgC) - 1], shape)

            if is_within_boundary:
                cute.copy(copy_atom, tXgX, rX)
                cute.copy(copy_atom, rX, tXgY)
            else:
                cute.copy(copy_atom, tXgX, rX, pred=rC)
                cute.copy(copy_atom, rX, tXgY, pred=rC)

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gY: cute.Tensor,
        gC: cute.Tensor,
        gCu_seqlens: cute.Tensor,
        copy_atom: cute.CopyAtom,
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID_S, BLOCK_ID_B, _ = cute.arch.block_idx()

        start = gCu_seqlens[BLOCK_ID_B]
        end = gCu_seqlens[BLOCK_ID_B + 1]
        seqlens = end - start

        S = cute.size(gX if const_expr(self.pack) else gY, mode=[1])

        pad_tokens = (S - seqlens) if const_expr(self.padding_side == "left") else 0
        t = start + BLOCK_ID_S - pad_tokens

        if const_expr(self.pack):
            bXgX = gX[BLOCK_ID_B, BLOCK_ID_S, None]
            bXgC = gC[BLOCK_ID_B, BLOCK_ID_S, None]
            bXgY = gY[t, None]
        else:
            bXgX = gX[t, None]
            bXgC = gC[t, None]
            bXgY = gY[BLOCK_ID_B, BLOCK_ID_S, None]

        if const_expr(self.padding_side == "left"):
            if BLOCK_ID_S >= pad_tokens:
                self._copy_array(bXgX=bXgX, bXgY=bXgY, bXgC=bXgC, copy_atom=copy_atom, shape=shape)
        elif BLOCK_ID_S < seqlens:
            self._copy_array(bXgX=bXgX, bXgY=bXgY, bXgC=bXgC, copy_atom=copy_atom, shape=shape)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mY: cute.Tensor, mCu_seqlens: cute.Tensor, stream: cuda.CUstream) -> None:
        mC = cute.make_identity_tensor(mX.shape)
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mX.element_type)

        if const_expr(self.pack):
            B = cute.size(mX, mode=[0])
            S = cute.size(mX, mode=[1])
        else:
            B = cute.size(mY, mode=[0])
            S = cute.size(mY, mode=[1])

        kernel = self.kernel(gX=mX, gY=mY, gC=mC, gCu_seqlens=mCu_seqlens, copy_atom=copy_atom, shape=mX.shape)
        kernel.launch(grid=(S, B, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream)


_CACHE = {}


@xma_op(mutates_args={"y"})
def _pack_unpack_sequence_cuda(
    x: torch.Tensor, y: torch.Tensor, cu_seqlens: torch.Tensor, padding_side: str, pack: bool, BLOCK_SIZE: int
) -> None:
    if pack:
        x = x.flatten(2, -1)
        y = y.flatten(1, -1)
        B, _, N = x.size()
    else:
        x = x.flatten(1, -1)
        y = y.flatten(2, -1)
        B, _, N = y.size()

    key = (x.dtype, N, pack, padding_side, BLOCK_SIZE)
    function = _CACHE.get(key, None)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if function is None:
        x_div = math.gcd(16 // x.dtype.itemsize, N)
        cu_seqlens_div = math.gcd(16 // cu_seqlens.dtype.itemsize, B + 1)

        _x = get_fake_cute_tensor(dtype=x.dtype, shape=(cute.sym_int(), cute.sym_int(), N), divisibility=x_div)
        _y = get_fake_cute_tensor(dtype=x.dtype, shape=(cute.sym_int(), N), divisibility=x_div)
        _cu_seqlens = get_fake_cute_tensor(
            dtype=cu_seqlens.dtype, shape=(cute.sym_int(),), divisibility=cu_seqlens_div
        )

        if not pack:
            _x, _y = _y, _x

        function = _PackUnpackSequenceCUDAKernel(N=N, padding_side=padding_side, pack=pack, BLOCK_SIZE=BLOCK_SIZE)

        function = cute.compile(function, _x, _y, _cu_seqlens, stream, options="--enable-tvm-ffi")
        _CACHE[key] = function

    function(x, y, cu_seqlens, stream)


class _PackSequenceCUDA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
    ) -> torch.Tensor:
        x = x.contiguous()
        cu_seqlens = cu_seqlens.contiguous()

        ctx_save_for_backward(ctx, cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()

        y = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        _pack_unpack_sequence_cuda(
            x=x, y=y, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=True, BLOCK_SIZE=1024
        )

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        cu_seqlens = ctx.saved_tensors[0]
        dy = dy.contiguous()

        dx = torch.zeros(*ctx.x_shape, device=dy.device, dtype=dy.dtype)
        _pack_unpack_sequence_cuda(
            x=dy, y=dx, cu_seqlens=cu_seqlens, padding_side=ctx.padding_side, pack=False, BLOCK_SIZE=1024
        )

        return dx, None, None, None


class _UnpackSequenceCUDA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
    ) -> torch.Tensor:
        x = x.contiguous()
        cu_seqlens = cu_seqlens.contiguous()

        ctx_save_for_backward(ctx, cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()

        y = torch.zeros(*output_shape, device=x.device, dtype=x.dtype)
        _pack_unpack_sequence_cuda(
            x=x, y=y, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=False, BLOCK_SIZE=1024
        )

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        cu_seqlens = ctx.saved_tensors[0]
        dy = dy.contiguous()

        dx = torch.empty(ctx.x_shape, device=dy.device, dtype=dy.dtype)
        _pack_unpack_sequence_cuda(
            x=dy, y=dx, cu_seqlens=cu_seqlens, padding_side=ctx.padding_side, pack=True, BLOCK_SIZE=1024
        )

        return dx, None, None, None
