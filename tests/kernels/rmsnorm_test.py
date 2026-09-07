# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Callable

import pytest


torch = pytest.importorskip("torch")

from lm_engine.kernels import Accelerator, KernelBackend
from lm_engine.kernels.functional import rmsnorm
from tests.kernels.fused_residual_add_rmsnorm_test import _get_sizes
from tests.layers.utils import assert_equal_tensors, get_duplicated_tensors, skip_if_incompatible_kernel_backend


_EPSILON = 1e-5
_SEED = 42


@pytest.mark.parametrize("size", _get_sizes())
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("memory_efficient", [False, True])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("function", [rmsnorm, torch.compile(rmsnorm, fullgraph=True)])
@torch._dynamo.config.patch(recompile_limit=1024)
def test_rmsnorm(
    size: tuple[int],
    kernel_backend: KernelBackend,
    dtype: torch.dtype,
    memory_efficient: bool,
    has_weight: bool,
    function: Callable,
) -> None:
    device = skip_if_incompatible_kernel_backend(kernel_backend)

    Accelerator.set_seed(_SEED)

    x_kernel, x_expected = get_duplicated_tensors(size, device=device, dtype=dtype, std=None)

    if has_weight:
        weight_kernel, weight_expected = get_duplicated_tensors((size[-1],), device=device, dtype=dtype, std=None)
    else:
        weight_kernel = None
        weight_expected = None

    z_kernel = function(
        x=x_kernel,
        weight=weight_kernel,
        eps=_EPSILON,
        memory_efficient=memory_efficient,
        kernel_backend=kernel_backend,
    )

    z_expected = rmsnorm(x=x_expected, weight=weight_expected, eps=_EPSILON, kernel_backend=KernelBackend.torch)

    z_kernel.sum().backward()
    z_expected.sum().backward()

    assert_equal_tensors(z_kernel, z_expected, False, atol_float16=1.6e-2, rtol_float16=0)
    assert_equal_tensors(
        x_kernel.grad,
        x_expected.grad,
        False,
        atol_float32=1.2e-5,
        rtol_float32=0,
        atol_float16=9e-2,
        rtol_float16=0,
    )

    if has_weight:
        assert_equal_tensors(
            weight_kernel.grad,
            weight_expected.grad,
            False,
            atol_float32=6.5e-5,
            rtol_float32=0,
            atol_float16=0.1,
            rtol_float16=0.01,
        )
