# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch

from lm_engine.accelerator import KernelBackend
from lm_engine.utils import is_cute_dsl_available, is_triton_available
from tests.utils import skip_test_if_device_unavailable


_DEFAULT_TOLERANCES = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=1e-2, rtol=1e-2),
}

# package (not just hardware) availability required for each backend to actually have a kernel
# registered, on top of the accelerator-type check in KernelBackend.verify_accelerator()
_BACKEND_AVAILABILITY_CHECKS = {
    KernelBackend.triton: (is_triton_available, "triton"),
    KernelBackend.cuda: (is_cute_dsl_available, "cute_dsl (cutlass)"),
}


def skip_if_incompatible_kernel_backend(kernel_backend: KernelBackend) -> torch.device:
    if not kernel_backend.verify_accelerator():
        pytest.skip(f"skipping test because kernel_backend ({kernel_backend}) is incompatible with the accelerator")

    availability_check = _BACKEND_AVAILABILITY_CHECKS.get(kernel_backend)
    if availability_check is not None:
        is_available, name = availability_check
        if not is_available():
            pytest.skip(f"skipping test because {name} is unavailable")

    device = torch.device(kernel_backend.get_compatible_accelerator().value)
    skip_test_if_device_unavailable(device)

    return device


def get_duplicated_tensors(
    shape: tuple[int, ...], device: torch.device, dtype: torch.dtype, std: float = 0.01, requires_grad: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """2 leaf tensors with identical values but independent autograd graphs, so a kernel path and a
    torch path can be run (and backpropagated) without their gradients interfering."""

    base = torch.randn(shape, device=device, dtype=dtype) * std
    a = base.clone().requires_grad_(requires_grad)
    b = base.clone().requires_grad_(requires_grad)
    return a, b


def assert_equal_tensors(x: torch.Tensor, y: torch.Tensor, exact_match: bool, **kwargs) -> None:
    """compares x and y; pass exact_match=True to require bitwise equality, otherwise falls back to
    a per-dtype default tolerance that can be overridden per call with atol_<dtype>/rtol_<dtype>
    kwargs (e.g. atol_bfloat16=1e-2)."""

    if exact_match:
        assert torch.equal(x, y)
        return

    dtype_name = str(x.dtype).rsplit(".", 1)[-1]
    defaults = _DEFAULT_TOLERANCES[x.dtype]

    atol = kwargs.get(f"atol_{dtype_name}", defaults["atol"])
    rtol = kwargs.get(f"rtol_{dtype_name}", defaults["rtol"])

    torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
