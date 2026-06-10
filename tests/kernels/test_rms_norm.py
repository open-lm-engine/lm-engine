# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch
from torch.testing import assert_close

from lm_engine.accelerator import Accelerator
from lm_engine.arguments import KernelArgs
from lm_engine.enums import Kernel
from lm_engine.hf_models.modeling_utils.normalization import RMSNorm
from lm_engine.kernels import enable_kernels
from lm_engine.utils import is_quack_available, is_xma_available
from tests.utils import skip_test_if_device_unavailable


SEED = 1234
SHAPE = (64, 128)


def _copy_rmsnorm(module: RMSNorm, device: torch.device, dtype: torch.dtype) -> RMSNorm:
    copied = RMSNorm(module.normalized_shape[0], eps=module.eps).to(device=device, dtype=dtype)
    copied.load_state_dict(module.state_dict())
    return copied


def _run_rmsnorm(
    module: RMSNorm,
    x: torch.Tensor,
    grad: torch.Tensor,
    kernels: list[Kernel],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.detach().clone().requires_grad_(True)

    with enable_kernels(kernels):
        y = module(x)
        y.backward(grad)

    assert module.weight.grad is not None
    assert x.grad is not None

    return y.detach(), x.grad.detach(), module.weight.grad.detach()


@pytest.mark.parametrize("device", [torch.device("cuda")])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_xma_rmsnorm_equivalence(device: torch.device, dtype: torch.dtype) -> None:
    skip_test_if_device_unavailable(device)

    if not is_xma_available():
        pytest.skip("skipping test because accelerated-model-architectures is unavailable")

    Accelerator.set_seed(SEED)

    x = torch.randn(*SHAPE, device=device, dtype=dtype)
    grad = torch.randn_like(x)

    torch_module = RMSNorm(SHAPE[-1], eps=1e-5).to(device=device, dtype=dtype)
    xma_module = _copy_rmsnorm(torch_module, device, dtype)

    y_torch, dx_torch, dw_torch = _run_rmsnorm(torch_module, x, grad, [])
    y_xma, dx_xma, dw_xma = _run_rmsnorm(xma_module, x, grad, [Kernel.rmsnorm])

    assert_close(y_xma, y_torch, rtol=5e-3, atol=5e-3)
    assert_close(dx_xma, dx_torch, rtol=5e-3, atol=5e-3)
    assert_close(dw_xma, dw_torch, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("device", [torch.device("cuda")])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_xma_memory_efficient_rmsnorm_equivalence(device: torch.device, dtype: torch.dtype) -> None:
    skip_test_if_device_unavailable(device)

    if not is_xma_available():
        pytest.skip("skipping test because accelerated-model-architectures is unavailable")

    Accelerator.set_seed(SEED)

    x = torch.randn(*SHAPE, device=device, dtype=dtype)
    grad = torch.randn_like(x)

    torch_module = RMSNorm(SHAPE[-1], eps=1e-5).to(device=device, dtype=dtype)
    xma_module = _copy_rmsnorm(torch_module, device, dtype)

    y_torch, dx_torch, dw_torch = _run_rmsnorm(torch_module, x, grad, [])
    y_xma, dx_xma, dw_xma = _run_rmsnorm(xma_module, x, grad, [Kernel.rmsnorm_memory_efficient])

    assert_close(y_xma, y_torch, rtol=5e-3, atol=5e-3)
    assert_close(dx_xma, dx_torch, rtol=5e-3, atol=5e-3)
    assert_close(dw_xma, dw_torch, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("device", [torch.device("cuda")])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quack_rmsnorm_equivalence(device: torch.device, dtype: torch.dtype) -> None:
    skip_test_if_device_unavailable(device)

    if not is_quack_available():
        pytest.skip("skipping test because quack-kernels is unavailable")

    Accelerator.set_seed(SEED)

    x = torch.randn(*SHAPE, device=device, dtype=dtype)
    grad = torch.randn_like(x)

    torch_module = RMSNorm(SHAPE[-1], eps=1e-5).to(device=device, dtype=dtype)
    quack_module = _copy_rmsnorm(torch_module, device, dtype)

    y_torch, dx_torch, dw_torch = _run_rmsnorm(torch_module, x, grad, [])
    y_quack, dx_quack, dw_quack = _run_rmsnorm(quack_module, x, grad, [Kernel.quack_rmsnorm])

    assert_close(y_quack, y_torch, rtol=5e-3, atol=5e-3)
    assert_close(dx_quack, dx_torch, rtol=5e-3, atol=5e-3)
    assert_close(dw_quack, dw_torch, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("xma_kernel", [Kernel.rmsnorm, Kernel.rmsnorm_memory_efficient])
def test_quack_rmsnorm_is_mutually_exclusive_with_xma(xma_kernel: Kernel) -> None:
    with pytest.raises(ValueError, match="quack_rmsnorm cannot be enabled with XMA RMSNorm"):
        KernelArgs(kernels=[Kernel.quack_rmsnorm, xma_kernel])


def test_quack_rmsnorm_rejects_tensor_parallel() -> None:
    if not is_quack_available():
        pytest.skip("skipping test because quack-kernels is unavailable")

    module = RMSNorm(SHAPE[-1], eps=1e-5)
    module.is_tp_enabled = True

    with enable_kernels([Kernel.quack_rmsnorm]):
        with pytest.raises(AssertionError, match="does not support tensor parallel"):
            module(torch.randn(*SHAPE))


@pytest.mark.parametrize("kernel", [Kernel.rmsnorm, Kernel.rmsnorm_memory_efficient])
def test_xma_rmsnorm_rejects_tensor_parallel(kernel: Kernel) -> None:
    if not is_xma_available():
        pytest.skip("skipping test because accelerated-model-architectures is unavailable")

    module = RMSNorm(SHAPE[-1], eps=1e-5)
    module.is_tp_enabled = True

    with enable_kernels([kernel]):
        with pytest.raises(AssertionError, match="does not support tensor parallel"):
            module(torch.randn(*SHAPE))
