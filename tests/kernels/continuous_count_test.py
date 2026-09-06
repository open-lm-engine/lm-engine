# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Callable

import pytest


torch = pytest.importorskip("torch")

from lm_engine.accelerator import Accelerator, KernelBackend
from lm_engine.functional import continuous_count
from tests.layers.utils import assert_equal_tensors, get_1d_tensor_sizes, skip_if_incompatible_kernel_backend


_MAX_EXPERTS = 72
_SEED = 42


@pytest.mark.parametrize("bins", get_1d_tensor_sizes())
@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda])
@pytest.mark.parametrize("dtype", [torch.long, torch.int])
@pytest.mark.parametrize("function", [continuous_count, torch.compile(continuous_count, fullgraph=True)])
@torch._dynamo.config.patch(recompile_limit=1024)
def test_continuous_count(
    bins: int,
    kernel_backend: KernelBackend,
    dtype: torch.dtype,
    function: Callable,
) -> None:
    device = skip_if_incompatible_kernel_backend(kernel_backend)

    Accelerator.set_seed(_SEED)
    x = torch.randint(0, _MAX_EXPERTS, (bins,), device=device, dtype=dtype)

    z_kernel = function(x=x, bins=_MAX_EXPERTS, kernel_backend=kernel_backend)
    z_expected = continuous_count(x.view(-1), bins=_MAX_EXPERTS, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(z_kernel, z_expected, True)
