# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp
from ...utils import is_triton_available
from .torch_implementation import _softmax_torch


class _Softmax(CustomOp): ...


_Softmax[KernelBackend.torch] = _softmax_torch


if is_triton_available():
    from .triton_implementation import _SoftmaxTriton

    _Softmax[KernelBackend.cuda] = _SoftmaxTriton
    _Softmax[KernelBackend.triton] = _SoftmaxTriton


def softmax(
    x: torch.Tensor, logits_multiplier: float | None = None, *, kernel_backend: KernelBackend | None = None
) -> torch.Tensor:
    """
    computes softmax activation

    :param x: input activation tensor
    :type x: torch.Tensor
    :param logits_multiplier: pre-multiplies `x` with `logits_multiplier` before computing softmax.
        Defaults to None.
    :type logits_multiplier: float | None
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor
    :rtype: Tensor
    """

    return _Softmax.run(x=x, logits_multiplier=logits_multiplier, kernel_backend=kernel_backend)
