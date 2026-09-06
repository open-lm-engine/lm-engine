# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import random
from typing import Callable

import pytest
import torch

from lm_engine.kernels import Accelerator, KernelBackend
from lm_engine.kernels.functional import fused_linear_cross_entropy
from tests.layers.utils import (
    assert_equal_tensors,
    get_2d_tensor_sizes,
    get_duplicated_tensors,
    skip_if_incompatible_kernel_backend,
)


_SEED = 42


@pytest.mark.parametrize("size", get_2d_tensor_sizes())
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("logits_multiplier", [None, 0.7])
@pytest.mark.parametrize(
    "function",
    [
        fused_linear_cross_entropy,
        torch.compile(fused_linear_cross_entropy, fullgraph=True),
    ],
)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_fused_linear_cross_entropy(
    size: tuple[int],
    kernel_backend: KernelBackend,
    dtype: torch.dtype,
    logits_multiplier: float | None,
    function: Callable,
) -> None:
    device = skip_if_incompatible_kernel_backend(kernel_backend)

    Accelerator.set_seed(_SEED)

    if isinstance(size, int):
        size = (size,)

    x_kernel, x_expected = get_duplicated_tensors(size, device=device, dtype=dtype, std=0.02)

    vocab_size = random.randint(max(100, size[0] - 100), size[0] + 100)
    weight_kernel, weight_expected = get_duplicated_tensors(
        (vocab_size, size[1]), device=device, dtype=dtype, std=0.02
    )

    labels = torch.randint(0, vocab_size, (x_kernel.size(0),), device=x_kernel.device)

    loss_kernel = function(
        x=x_kernel,
        weight=weight_kernel,
        labels=labels,
        logits_multiplier=logits_multiplier,
        kernel_backend=kernel_backend,
    )

    loss_expected = fused_linear_cross_entropy(
        x=x_expected,
        weight=weight_expected,
        labels=labels,
        logits_multiplier=logits_multiplier,
        kernel_backend=KernelBackend.torch,
    )

    loss_kernel.backward()
    loss_expected.backward()

    assert_equal_tensors(loss_kernel, loss_expected, False, atol_float32=5.1e-5, rtol_float32=0)
    assert_equal_tensors(x_kernel.grad, x_expected.grad, False)

    assert_equal_tensors(
        weight_kernel.grad,
        weight_expected.grad,
        False,
        atol_float32=5.2e-5,
        rtol_float32=0,
        atol_bfloat16=6.2e-5,
        rtol_bfloat16=0,
    )
