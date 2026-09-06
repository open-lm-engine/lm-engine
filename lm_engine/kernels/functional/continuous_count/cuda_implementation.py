# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from ...custom_op import xma_op
from ...cute_dsl_utils import get_fake_cute_tensor
from ...math import ceil_divide


class _ContinuousCountCUDAKernel:
    def __init__(self, B: int, C: int, BLOCK_SIZE: int = 128) -> _ContinuousCountCUDAKernel:
        self.B = min(B, 4)
        self.C = C
        self.BLOCK_SIZE = BLOCK_SIZE
        self.NUM_STORE_LOOPS = ceil_divide(C, BLOCK_SIZE)

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gY: cute.Tensor,
        gC: cute.Tensor,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()
        block_coord = ((None, None), BLOCK_ID)

        bX = gX[block_coord]
        bC = gC[block_coord]

        thr_copy = tiled_copy.get_slice(THREAD_ID)

        tX = thr_copy.partition_S(bX)
        tC = thr_copy.partition_S(bC)

        rX = cute.make_rmem_tensor_like(tX)
        rC = cute.make_rmem_tensor_like(tC, dtype=cutlass.Boolean)
        for i in cutlass.range_constexpr(cute.size(rC)):
            rC[i] = cute.elem_less(tC[i], shape)

        is_within_boundary = cute.elem_less(tC[cute.size(tC) - 1], shape)

        if is_within_boundary:
            cute.copy(copy_atom, tX, rX)
        else:
            cute.copy(copy_atom, tX, rX, pred=rC)

        sY = self._get_shared_memory(gY.element_type, THREAD_ID)

        for i in cutlass.range_constexpr(cute.size(rX)):
            xi = rX[i]
            if is_within_boundary or rC[i]:
                cute.arch.atomic_add(sY.iterator + xi, 1, sem="relaxed")

        cute.arch.sync_threads()

        self._write_out_output(sY=sY, gY=gY, BLOCK_ID=BLOCK_ID, THREAD_ID=THREAD_ID)

    @cute.jit
    def _get_shared_memory(self, dtype: cute.Numeric, THREAD_ID: int) -> cute.Tensor:
        elements_per_thread = 128 // dtype.width
        smem_allocator = cutlass.utils.SmemAllocator()

        shape = (1, self.C)
        val_shape = (1, elements_per_thread)

        sY = smem_allocator.allocate_tensor(
            dtype, layout=cute.make_ordered_layout(shape, order=(1, 0)), byte_alignment=16
        )

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype)

        rY = cute.make_rmem_tensor(cute.make_ordered_layout(val_shape, order=(1, 0)), sY.element_type)
        rY.fill(0)

        gC = cute.make_identity_tensor(shape)

        for i in cutlass.range_constexpr(self.NUM_STORE_LOOPS):
            idx = i * self.BLOCK_SIZE + THREAD_ID
            coord = (0, idx)
            tC = cute.local_tile(gC, val_shape, coord)

            rC = cute.make_rmem_tensor_like(rY, dtype=cutlass.Boolean)
            for j in cutlass.range_constexpr(cute.size(rC)):
                rC[j] = cute.elem_less(tC[j], shape)

            is_within_boundary = cute.elem_less(tC[cute.size(tC) - 1], shape)

            if is_within_boundary:
                cute.copy(copy_atom, rY, cute.local_tile(sY, val_shape, coord))
            else:
                cute.copy(copy_atom, rY, cute.local_tile(sY, val_shape, coord), pred=rC)

        cute.arch.sync_threads()

        return sY

    @cute.jit
    def _write_out_output(self, sY: cute.Tensor, gY: cute.Tensor, BLOCK_ID: int, THREAD_ID: int) -> None:
        dtype = sY.element_type
        elements_per_thread = 128 // dtype.width

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype)

        shape = (1, self.C)
        val_shape = (1, elements_per_thread)

        gC = cute.make_identity_tensor(shape)

        for i in cutlass.range_constexpr(self.NUM_STORE_LOOPS):
            idx = i * self.BLOCK_SIZE + THREAD_ID
            coord = (0, idx)
            tC = cute.local_tile(gC, val_shape, coord)

            rY = cute.make_rmem_tensor(cute.make_ordered_layout(val_shape, order=(1, 0)), sY.element_type)
            rC = cute.make_rmem_tensor_like(rY, dtype=cutlass.Boolean)
            for j in cutlass.range_constexpr(cute.size(rC)):
                rC[j] = cute.elem_less(tC[j], shape)

            is_within_boundary = cute.elem_less(tC[cute.size(tC) - 1], shape)

            if is_within_boundary:
                cute.copy(copy_atom, cute.local_tile(sY, val_shape, coord), rY)
            else:
                cute.copy(copy_atom, cute.local_tile(sY, val_shape, coord), rY, pred=rC)

            gidx = idx * elements_per_thread

            for j in cutlass.range_constexpr(elements_per_thread):
                if gidx < self.C:
                    cute.arch.atomic_add(gY.iterator + gidx, rY[j], sem="relaxed")

                gidx += 1

    @cute.jit
    def __call__(self, mX: cute.Tensor, mY: cute.Tensor, stream: cuda.CUstream) -> None:
        vector_size = 128 // mX.element_type.width

        thr_layout = cute.make_ordered_layout((1, self.BLOCK_SIZE), order=(1, 0))
        val_layout = cute.make_ordered_layout((self.B, vector_size), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        mC = cute.make_identity_tensor(mX.shape)

        gX = cute.zipped_divide(mX, tiler_mn)
        gC = cute.zipped_divide(mC, tiler_mn)

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        NUM_BLOCKS = cute.size(gX, mode=[1])
        kernel = self.kernel(gX=gX, gY=mY, gC=gC, copy_atom=copy_atom, tiled_copy=tiled_copy, shape=mX.shape)
        kernel.launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream)


_CACHE = {}


@xma_op(mutates_args={"y"})
def _continuous_count_cuda(x: torch.Tensor, y: torch.Tensor) -> None:
    N = x.numel()
    C = y.numel()

    x_div = math.gcd(16 // x.dtype.itemsize, N)
    y_div = math.gcd(16 // y.dtype.itemsize, C)

    key = (x.dtype, y.dtype, C, x_div, y_div)
    function = _CACHE.get(key, None)

    if x.dim() == 1:
        x = x[None, ...]

    B = x.size(0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if function is None:
        _x = get_fake_cute_tensor(dtype=x.dtype, shape=(B, cute.sym_int()), divisibility=x_div)
        _y = get_fake_cute_tensor(dtype=y.dtype, shape=(C,), divisibility=y_div)

        function = _ContinuousCountCUDAKernel(B=B, C=C)

        function = cute.compile(function, _x, _y, stream, options="--enable-tvm-ffi")
        _CACHE[key] = function

    function(x, y, stream)
