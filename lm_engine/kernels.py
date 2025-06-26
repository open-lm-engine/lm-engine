# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from contextlib import contextmanager

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from .enums import Kernel
from .utils import is_cute_kernels_available


if is_cute_kernels_available():
    from cute_kernels.utils import get_boolean_env_variable

    _ENABLE_ALL_KERNELS = get_boolean_env_variable("ENABLE_ALL_CUTE_KERNELS", False)
else:
    _ENABLE_ALL_KERNELS = False

_KERNELS: set[Kernel] = set()


def is_kernel_allowed(kernel: Kernel) -> bool:
    return kernel in _KERNELS


@contextmanager
def enable_kernels(kernels: set[Kernel] | list[Kernel]):
    if not isinstance(kernels, set):
        kernels = set(kernels)

    global _KERNELS

    original_kernels = _KERNELS
    _KERNELS = kernels

    yield

    _KERNELS = original_kernels


@contextmanager
def enable_all_kernels():
    all_kernels = filter(lambda k: k != Kernel.ladder_residual_overlapped_layer, Kernel)
    all_kernels = list(all_kernels)

    with enable_kernels(all_kernels):
        yield


class _ACT_BackwardWait(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, x_grad: AsyncCollectiveTensor) -> torch.Tensor:
        if isinstance(x_grad, AsyncCollectiveTensor):
            x_grad = x_grad.wait()

        return x_grad


def wait_for_ACT(x: torch.Tensor, wait_in_forward: bool, wait_in_backward: bool) -> torch.Tensor:
    if wait_in_forward and isinstance(x, AsyncCollectiveTensor):
        x = x.wait()

    if wait_in_backward:
        x = _ACT_BackwardWait.apply(x)

    return x


if _ENABLE_ALL_KERNELS:
    enable_all_kernels().__enter__()
