# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import triton
import triton.language as tl


@triton.jit
def matmul(A, B, C, output_dtype: tl.constexpr):
    if A.shape[0] == 1:
        x = tl.sum(A.T * B, axis=0, keep_dims=True)
        if C is not None:
            x += C
        x = x.to(output_dtype)
    elif A.shape[1] == 1:
        x = A * B
        if C is not None:
            x += C
        x = x.to(output_dtype)
    elif B.shape[1] == 1:
        x = tl.sum(A * B.T, axis=1, keep_dims=True)
        if C is not None:
            x += C
        x = x.to(output_dtype)
    elif C is None:
        if output_dtype == tl.bfloat16:
            x = tl.dot(A, B, out_dtype=tl.float32).to(output_dtype)
        else:
            x = tl.dot(A, B, out_dtype=output_dtype)
    elif C.shape[0] == 1 or C.shape[1] == 1:
        x = tl.dot(A, B, out_dtype=tl.float32)
        x += C
        x = x.to(output_dtype)
    else:
        x = tl.dot(A, B, C.to(tl.float32), out_dtype=tl.float32).to(output_dtype)

    return x
