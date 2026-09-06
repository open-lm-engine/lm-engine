# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....accelerator import KernelBackend
from ....custom_op import CustomOp
from ....utils import is_cute_dsl_available
from .torch_implementation import _swiglu_packed_torch


class _SwigluPacked(CustomOp): ...


_SwigluPacked[KernelBackend.torch] = _swiglu_packed_torch


if is_cute_dsl_available():
    from .cuda_implementation import _SwigluPackedCUDA

    _SwigluPacked[KernelBackend.cuda] = _SwigluPackedCUDA


def swiglu_packed(x: torch.Tensor, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """
    computes swiglu activation by splitting the tensor `x` into 2 parts: gate and up activations. The tensor has
    interleaved values of gate, up, gate, up, ...

    :param x: input activation
    :type x: torch.Tensor
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor
    :rtype: Tensor
    """

    original_shape = x.size()
    x = x.flatten(0, -2)

    y = _SwigluPacked.run(x=x, kernel_backend=kernel_backend)
    y = y.view(*original_shape[:-1], original_shape[-1] // 2)

    return y
