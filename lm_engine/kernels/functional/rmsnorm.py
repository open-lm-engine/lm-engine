# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ..accelerator import KernelBackend
from .fused_residual_add_rmsnorm import fused_residual_add_rmsnorm


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None = None,
    memory_efficient: bool = False,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """
    RMSNorm computation

    :param x: input activation
    :type x: torch.Tensor
    :param weight: RMSNorm weight
    :type weight: torch.Tensor | None
    :param eps: epsilon. Defaults to None.
    :type eps: float | None
    :param memory_efficient: memory efficient = False caches RMSNorm's denominator in the forward.
        Defaults to False.
    :type memory_efficient: bool
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor
    :rtype: Tensor
    """

    x, _ = fused_residual_add_rmsnorm(
        x=x,
        residual=None,
        weight=weight,
        eps=eps,
        multiplier=None,
        memory_efficient=memory_efficient,
        kernel_backend=kernel_backend,
    )

    return x
