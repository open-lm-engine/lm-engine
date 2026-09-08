# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch

from lm_engine.kernels import Accelerator, KernelBackend
from lm_engine.training.optimization.adam_hyperball.op import adam_hyperball
from tests.layers.utils import assert_equal_tensors, get_1d_tensor_sizes, skip_if_incompatible_kernel_backend


_LEARNING_RATE = 1e-3
_SEED = 42


@pytest.mark.parametrize("size", get_1d_tensor_sizes())
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
def test_adam_hyperball(
    size: int,
    dtype: torch.dtype,
    maximize: bool,
    steps: int,
    kernel_backend: KernelBackend,
) -> None:
    """mirrors xma's own tests/optimizers/adam_hyperball_test.py, ported to call the vendored
    adam_hyperball op directly (instead of AdamHyperball.step, which doesn't expose kernel_backend)."""

    device = skip_if_incompatible_kernel_backend(kernel_backend)

    Accelerator.set_seed(_SEED)

    params_kernel = [torch.randint(-8, 8, (size,), device=device, dtype=dtype) for _ in range(3)]
    params_torch = [p.clone() for p in params_kernel]
    Rs = [p.norm() for p in params_kernel]

    grads = [torch.randint(-8, 8, (size,), device=device, dtype=dtype) for _ in range(3)]

    exp_avgs_kernel = [torch.zeros_like(p) for p in params_kernel]
    exp_avg_sqs_kernel = [torch.zeros_like(p) for p in params_kernel]
    exp_avgs_torch = [torch.zeros_like(p) for p in params_torch]
    exp_avg_sqs_torch = [torch.zeros_like(p) for p in params_torch]

    state_steps_kernel = [1, 1, 1]
    state_steps_torch = [1, 1, 1]

    for _ in range(steps):
        adam_hyperball(
            params=params_kernel,
            grads=grads,
            exp_avgs=exp_avgs_kernel,
            exp_avg_sqs=exp_avg_sqs_kernel,
            Rs=Rs,
            lr=_LEARNING_RATE,
            beta1=0.9,
            beta2=0.95,
            maximize=maximize,
            state_steps=state_steps_kernel,
            kernel_backend=kernel_backend,
        )

        adam_hyperball(
            params=params_torch,
            grads=grads,
            exp_avgs=exp_avgs_torch,
            exp_avg_sqs=exp_avg_sqs_torch,
            Rs=Rs,
            lr=_LEARNING_RATE,
            beta1=0.9,
            beta2=0.95,
            maximize=maximize,
            state_steps=state_steps_torch,
            kernel_backend=KernelBackend.torch,
        )

    for param_kernel, param_torch in zip(params_kernel, params_torch):
        assert_equal_tensors(param_kernel, param_torch, exact_match=False)
