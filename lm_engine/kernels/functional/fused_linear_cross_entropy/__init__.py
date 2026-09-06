# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp
from ...utils import is_triton_available
from .torch_implementation import _fused_linear_cross_entropy_torch


class _FusedLinearCrossEntropy(CustomOp): ...


_FusedLinearCrossEntropy[KernelBackend.torch] = _fused_linear_cross_entropy_torch


if is_triton_available():
    from .triton_implementation import _FusedLinearCrossEntropyTriton

    _FusedLinearCrossEntropy[KernelBackend.cuda] = _FusedLinearCrossEntropyTriton
    _FusedLinearCrossEntropy[KernelBackend.rocm] = _FusedLinearCrossEntropyTriton
    _FusedLinearCrossEntropy[KernelBackend.triton] = _FusedLinearCrossEntropyTriton


def fused_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    logits_multiplier: float | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """
    compute cross entropy loss without materializing the full output logits matrix

    :param x: logits
    :type x: torch.Tensor
    :param weight: vocab weight
    :type weight: torch.Tensor
    :param labels: labels
    :type labels: torch.Tensor
    :param reduction: reduction should be either sum or mean. Defaults to "mean".
    :type reduction: str
    :param logits_multiplier: logits multiplier pre-multiplies logits, None implies 1.
        Defaults to None.
    :type logits_multiplier: float | None
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: loss
    :rtype: Tensor
    """

    assert reduction in ["sum", "mean"]
    assert x.dim() == 2, "x should be 2 dimensional"
    assert labels.dim() == 1, "labels should be 1 dimensional"
    assert x.size(0) == labels.size(0), "x and labels have different number of elements along dim 0"
    assert x.size(-1) == weight.size(-1)

    x = _FusedLinearCrossEntropy.run(
        x=x,
        W=weight,
        y=labels,
        reduction=reduction,
        logits_multiplier=logits_multiplier,
        kernel_backend=kernel_backend,
    )

    return x
