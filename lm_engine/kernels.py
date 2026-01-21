# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os
from contextlib import contextmanager

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from .enums import Kernel


_ENABLE_ALL_KERNELS = os.getenv("ENABLE_ALL_KERNELS", "False").lower() in ["1", "true"]
_ENABLE_KERNELS = os.getenv("ENABLE_KERNELS", "")
_KERNELS = {kernel: False for kernel in Kernel}


if _ENABLE_ALL_KERNELS:
    assert not _ENABLE_KERNELS
elif _ENABLE_KERNELS:
    assert not _ENABLE_ALL_KERNELS


def is_kernel_allowed(kernel: Kernel) -> bool:
    return _KERNELS[kernel]


@contextmanager
def enable_kernels(kernels: list[Kernel]):
    global _KERNELS

    original_kernels = _KERNELS

    _KERNELS = {}
    for kernel in Kernel:
        _KERNELS[kernel] = kernel in kernels

    yield

    _KERNELS = original_kernels


@contextmanager
def enable_all_kernels():
    with enable_kernels(list(Kernel)):
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
