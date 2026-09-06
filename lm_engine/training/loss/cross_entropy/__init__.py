# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp
from ...utils import is_triton_available
from .torch_implementation import _cross_entropy_torch


class _CrossEntropy(CustomOp): ...


_CrossEntropy[KernelBackend.torch] = _cross_entropy_torch


if is_triton_available():
    from .triton_implementation import _CrossEntropyTriton

    _CrossEntropy[KernelBackend.cuda] = _CrossEntropyTriton
    _CrossEntropy[KernelBackend.triton] = _CrossEntropyTriton


def cross_entropy(
    x: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    logits_multiplier: float | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """
    cross entropy loss

    :param x: logits
    :type x: torch.Tensor
    :param labels: labels
    :type labels: torch.Tensor
    :param reduction: reduction method: "sum", "mean" or None
    :type reduction: str
    :param logits_multiplier: logits multiplier pre-multiplies logits, None implies 1. Defaults to None.
    :type logits_multiplier: float | None
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: loss
    :rtype: Tensor
    """

    assert reduction in ["sum", "mean"]
    assert x.dim() == 2, "x should be 2 dimensional"
    assert labels.dim() == 1, "labels should be 1 dimensional"
    assert labels.size(0) == x.size(0), "x and labels have different number of elements along batch dimension"

    x = _CrossEntropy.run(
        x=x, labels=labels, reduction=reduction, logits_multiplier=logits_multiplier, kernel_backend=kernel_backend
    )

    return x
