# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch

from lm_engine.accelerator import KernelBackend
from lm_engine.math import ceil_divide
from lm_engine.modeling_utils.activations import swiglu, swiglu_packed
from tests.layers.utils import assert_equal_tensors, get_duplicated_tensors, skip_if_incompatible_kernel_backend


_SHAPES = [(4, 8), (7, 15), (32, 256), (17, 4103)]
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_SEED = 42


def _align_last_dim(shape: tuple[int, int], dtype: torch.dtype, kernel_backend: KernelBackend, packed: bool) -> int:
    # the cute_dsl (cuda) kernel does vectorized loads/stores and needs the last dim aligned to
    # 16 bytes; swiglu_packed's last dim additionally must be even (interleaved gate/up)
    multiple = 2 if packed else 1
    if kernel_backend == KernelBackend.cuda:
        multiple *= 16 // dtype.itemsize

    return ceil_divide(shape[-1], multiple) * multiple


@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda, KernelBackend.triton])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_swiglu_forward_kernel_vs_torch(
    dtype: torch.dtype, shape: tuple[int, int], kernel_backend: KernelBackend
) -> None:
    device = skip_if_incompatible_kernel_backend(kernel_backend)
    torch.manual_seed(_SEED)

    shape = (shape[0], _align_last_dim(shape, dtype, kernel_backend, packed=False))
    g_kernel, g_torch = get_duplicated_tensors(shape, device=device, dtype=dtype)
    u_kernel, u_torch = get_duplicated_tensors(shape, device=device, dtype=dtype)

    y_kernel = swiglu(g_kernel, u_kernel, kernel_backend=kernel_backend)
    y_torch = swiglu(g_torch, u_torch, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(y_kernel, y_torch, False)


@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda, KernelBackend.triton])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_swiglu_backward_kernel_vs_torch(
    dtype: torch.dtype, shape: tuple[int, int], kernel_backend: KernelBackend
) -> None:
    device = skip_if_incompatible_kernel_backend(kernel_backend)
    torch.manual_seed(_SEED)

    shape = (shape[0], _align_last_dim(shape, dtype, kernel_backend, packed=False))
    g_kernel, g_torch = get_duplicated_tensors(shape, device=device, dtype=dtype)
    u_kernel, u_torch = get_duplicated_tensors(shape, device=device, dtype=dtype)

    y_kernel = swiglu(g_kernel, u_kernel, kernel_backend=kernel_backend)
    y_torch = swiglu(g_torch, u_torch, kernel_backend=KernelBackend.torch)

    y_kernel.sum().backward()
    y_torch.sum().backward()

    assert_equal_tensors(y_kernel, y_torch, False)
    assert_equal_tensors(g_kernel.grad, g_torch.grad, False)
    assert_equal_tensors(u_kernel.grad, u_torch.grad, False)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_swiglu_compiled_kernel_vs_torch(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.cuda)
    torch.manual_seed(_SEED)

    shape = (shape[0], _align_last_dim(shape, dtype, KernelBackend.cuda, packed=False))
    g_kernel, g_torch = get_duplicated_tensors(shape, device=device, dtype=dtype)
    u_kernel, u_torch = get_duplicated_tensors(shape, device=device, dtype=dtype)

    swiglu_compiled = torch.compile(swiglu, fullgraph=True)
    y_kernel = swiglu_compiled(g_kernel, u_kernel, kernel_backend=KernelBackend.cuda)
    y_torch = swiglu(g_torch, u_torch, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(y_kernel, y_torch, False)


# swiglu_packed only has a CUDA (cute_dsl) backend, no triton kernel exists for it


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_swiglu_packed_forward_kernel_vs_torch(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.cuda)
    torch.manual_seed(_SEED)

    rows, cols = shape
    cols = _align_last_dim((rows, cols * 2), dtype, KernelBackend.cuda, packed=True)
    x_kernel, x_torch = get_duplicated_tensors((rows, cols), device=device, dtype=dtype)

    y_kernel = swiglu_packed(x_kernel, kernel_backend=KernelBackend.cuda)
    y_torch = swiglu_packed(x_torch, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(y_kernel, y_torch, False)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_swiglu_packed_backward_kernel_vs_torch(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.cuda)
    torch.manual_seed(_SEED)

    rows, cols = shape
    cols = _align_last_dim((rows, cols * 2), dtype, KernelBackend.cuda, packed=True)
    x_kernel, x_torch = get_duplicated_tensors((rows, cols), device=device, dtype=dtype)

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

    rows, cols = shape
    cols = _align_last_dim((rows, cols * 2), dtype, KernelBackend.cuda, packed=True)
    x_kernel, x_torch = get_duplicated_tensors((rows, cols), device=device, dtype=dtype)

    swiglu_packed_compiled = torch.compile(swiglu_packed, fullgraph=True)
    y_kernel = swiglu_packed_compiled(x_kernel, kernel_backend=KernelBackend.cuda)
    y_torch = swiglu_packed(x_torch, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(y_kernel, y_torch, False)
