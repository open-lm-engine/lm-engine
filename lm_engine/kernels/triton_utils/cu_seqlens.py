# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import triton
import triton.language as tl


@triton.jit
def get_start_end(cu_seqlens_ptr, cu_seqlens_stride, BLOCK_B, MASK_B):
    cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_B[:, None] * cu_seqlens_stride[0]
    start = tl.load(cu_seqlens_ptrs, mask=MASK_B[:, None])
    end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0], mask=MASK_B[:, None])

    return start, end
