# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Callable

import pytest


torch = pytest.importorskip("torch")

from lm_engine.kernels import Accelerator, KernelBackend
from lm_engine.kernels.functional import fused_residual_add_rmsnorm
from tests.layers.utils import (
    assert_equal_tensors,
    get_1d_tensor_sizes,
    get_duplicated_tensors,
    skip_if_incompatible_kernel_backend,
)


_EPSILON = 1e-5
_SEED = 42


def _get_sizes() -> list[tuple]:
    sizes = []
    for size in get_1d_tensor_sizes(max_offset=5):
        sizes.append((400, size))

    return sizes


@pytest.mark.parametrize("size", _get_sizes())
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("memory_efficient", [False, True])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("multiplier", [None, 0.9])
@pytest.mark.parametrize(
    "function",
    [
        fused_residual_add_rmsnorm,
        torch.compile(fused_residual_add_rmsnorm, fullgraph=True),
    ],
)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_fused_residual_add_rmsnorm(
    size: tuple[int] | int,
    kernel_backend: KernelBackend,
    dtype: torch.dtype,
    memory_efficient: bool,
    has_weight: bool,
    multiplier: float | None,
    function: Callable,
) -> None:
    device = skip_if_incompatible_kernel_backend(kernel_backend)

    Accelerator.set_seed(_SEED)

    x_kernel, x_expected = get_duplicated_tensors(size, device=device, dtype=dtype, std=None)
    residual_kernel, residual_expected = get_duplicated_tensors(size, device=device, dtype=dtype, std=None)

    if has_weight:
        weight_kernel, weight_expected = get_duplicated_tensors((size[-1],), device=device, dtype=dtype, std=None)
    else:
        weight_kernel = None
        weight_expected = None

    z_kernel, r_kernel = function(
        x=x_kernel,
        residual=residual_kernel,
        weight=weight_kernel,
        eps=_EPSILON,
        multiplier=multiplier,
        memory_efficient=memory_efficient,
        kernel_backend=kernel_backend,
    )

    z_expected, r_expected = fused_residual_add_rmsnorm(
        x=x_expected,
        residual=residual_expected,
        weight=weight_expected,
        eps=_EPSILON,
        multiplier=multiplier,
        kernel_backend=KernelBackend.torch,
    )

    z_kernel = z_kernel * 2 + r_kernel * 3
    z_expected = z_expected * 2 + r_expected * 3

    assert_equal_tensors(z_kernel, z_expected, False)

    z_kernel.sum().backward()
    z_expected.sum().backward()

    assert_equal_tensors(x_kernel.grad, x_expected.grad, False)
    assert_equal_tensors(residual_kernel.grad, residual_expected.grad, False)

    if has_weight:
        assert_equal_tensors(weight_kernel.grad, weight_expected.grad, False, atol_float32=7.7e-5, rtol_float32=0)
