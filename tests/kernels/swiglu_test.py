# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch

from lm_engine.kernels import KernelBackend
from lm_engine.kernels.functional import swiglu_packed
from lm_engine.math import ceil_divide
from tests.layers.utils import assert_equal_tensors, get_duplicated_tensors, skip_if_incompatible_kernel_backend


_SHAPES = [(4, 8), (7, 15), (32, 256), (17, 4103)]
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_SEED = 42


def _get_packed_shape(shape: tuple[int, int], dtype: torch.dtype) -> tuple[int, int]:
    # the cute_dsl (cuda) kernel does vectorized loads/stores and needs the last dim aligned to 16
    # bytes, and the interleaved gate/up layout needs it to be even on top of that
    multiple = 2 * (16 // dtype.itemsize)
    return (shape[0], ceil_divide(shape[-1] * 2, multiple) * multiple)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_swiglu_packed_forward_kernel_vs_torch(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.cuda)
    torch.manual_seed(_SEED)

    x_kernel, x_torch = get_duplicated_tensors(_get_packed_shape(shape, dtype), device=device, dtype=dtype)

    y_kernel = swiglu_packed(x_kernel, kernel_backend=KernelBackend.cuda)
    y_torch = swiglu_packed(x_torch, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(y_kernel, y_torch, False)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_swiglu_packed_backward_kernel_vs_torch(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.cuda)
    torch.manual_seed(_SEED)

    x_kernel, x_torch = get_duplicated_tensors(_get_packed_shape(shape, dtype), device=device, dtype=dtype)

    y_kernel = swiglu_packed(x_kernel, kernel_backend=KernelBackend.cuda)
    y_torch = swiglu_packed(x_torch, kernel_backend=KernelBackend.torch)

    y_kernel.sum().backward()
    y_torch.sum().backward()

    assert_equal_tensors(y_kernel, y_torch, False)
    assert_equal_tensors(x_kernel.grad, x_torch.grad, False)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_swiglu_packed_compiled_kernel_vs_torch(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.cuda)
    torch.manual_seed(_SEED)

    x_kernel, x_torch = get_duplicated_tensors(_get_packed_shape(shape, dtype), device=device, dtype=dtype)

    swiglu_packed_compiled = torch.compile(swiglu_packed, fullgraph=True)
    y_kernel = swiglu_packed_compiled(x_kernel, kernel_backend=KernelBackend.cuda)
    y_torch = swiglu_packed(x_torch, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(y_kernel, y_torch, False)
