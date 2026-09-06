# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp
from ...utils import is_triton_available
from .torch_implementation import _fused_residual_add_rmsnorm_torch


class _FusedResidualAddRMSNorm(CustomOp): ...


_FusedResidualAddRMSNorm[KernelBackend.torch] = _fused_residual_add_rmsnorm_torch


if is_triton_available():
    from .triton_implementation import _FusedResidualAddRMSNormTriton

    _FusedResidualAddRMSNorm[KernelBackend.cuda] = _FusedResidualAddRMSNormTriton
    _FusedResidualAddRMSNorm[KernelBackend.triton] = _FusedResidualAddRMSNormTriton


def fused_residual_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor | None,
    eps: float | None,
    multiplier: float | None = None,
    memory_efficient: bool = False,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    fused residual add RMSNorm computation

    :param x: input activation
    :type x: torch.Tensor
    :param residual: residual activation
    :type residual: torch.Tensor | None
    :param weight: RMSNorm weight
    :type weight: torch.Tensor | None
    :param eps: epsilon
    :type eps: float | None
    :param multiplier: if not None, pre-multiplies `x` with `multiplier`. Defaults to None.
    :type multiplier: float | None
    :param memory_efficient: memory efficient = False caches RMSNorm's denominator in the forward. Defaults to False.
    :type memory_efficient: bool
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output activations and updated residual stream
    :rtype: tuple[Tensor, Tensor | None]
    """

    if weight is not None:
        assert weight.dim() == 1, "weight should be 1D"
        assert x.dim() == 2

        if residual is not None:
            assert residual.dim() == 2

        assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
        assert weight.type() == x.type(), "tensors weight and y should have same dtype"

    x, residual = _FusedResidualAddRMSNorm.run(
        x=x,
        r=residual,
        W=weight,
        eps=eps,
        multiplier=multiplier,
        memory_efficient=memory_efficient,
        kernel_backend=kernel_backend,
    )

    return x, residual
