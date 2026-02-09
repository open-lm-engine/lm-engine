# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate

from ...dtensors import dtensor_to_tensor, tensor_to_dtensor
from ...enums import Kernel
from ...kernels import is_kernel_allowed, wait_for_ACT
from ...utils import ProcessGroupManager, is_xma_available
from ..parameter import mark_parameter_as_initialized, mark_parameter_as_no_weight_decay
from .dtensor_module import DTensorModule
from .TP import get_module_placements


if is_xma_available():
    from xma import rmsnorm


class LayerNorm(nn.LayerNorm, DTensorModule):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> LayerNorm:
        super().__init__(normalized_shape, eps=eps)

        self.is_tp_enabled = ProcessGroupManager.is_tensor_parallel_enabled()

        if self.is_tp_enabled:
            self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()
            self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

            self.weight = nn.Parameter(
                tensor_to_dtensor(self.weight, device_mesh=self.tp_mesh, current_placement=Replicate())
            )

            self.bias = nn.Parameter(
                tensor_to_dtensor(self.bias, device_mesh=self.tp_mesh, current_placement=Replicate())
            )

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_tp_enabled:
            x = tensor_to_dtensor(x, device_mesh=self.tp_mesh, current_placement=self.placement)

        x = super().forward(x)

        if self.is_tp_enabled:
            x = dtensor_to_tensor(x, device_mesh=self.tp_mesh, desired_placement=self.placement)

        return x

    def reset_parameters(self) -> None:
        super().reset_parameters()
        mark_parameter_as_initialized(self.weight)
        mark_parameter_as_initialized(self.bias)


class RMSNorm(nn.RMSNorm, DTensorModule):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> RMSNorm:
        super().__init__(normalized_shape, eps=eps)

        self.is_tp_enabled = ProcessGroupManager.is_tensor_parallel_enabled()

        if self.is_tp_enabled:
            self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()
            self.placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

            self.weight = nn.Parameter(
                tensor_to_dtensor(self.weight, device_mesh=self.tp_mesh, current_placement=Replicate())
            )

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_tp_enabled:
            x = tensor_to_dtensor(x, device_mesh=self.tp_mesh, current_placement=self.placement)

        if is_kernel_allowed(Kernel.rmsnorm) or is_kernel_allowed(Kernel.rmsnorm_memory_efficient):
            x = wait_for_ACT(x, wait_in_forward=True, wait_in_backward=False)

            x = rmsnorm(
                x=x,
                weight=self.weight,
                eps=self.eps,
                memory_efficient=is_kernel_allowed(Kernel.rmsnorm_memory_efficient),
            )

            x = wait_for_ACT(x, wait_in_forward=False, wait_in_backward=True)
        else:
            x = super().forward(x)

        if self.is_tp_enabled:
            x = dtensor_to_tensor(x, device_mesh=self.tp_mesh, desired_placement=self.placement)

        return x

    def reset_parameters(self) -> None:
        super().reset_parameters()
        mark_parameter_as_initialized(self.weight)


class PNorm(RMSNorm):
    def __init__(
        self,
        normalized_shape: int,
        p: int,
        eps: float | None = None,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> PNorm:
        self.p = p

        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_tp_enabled:
            x = tensor_to_dtensor(x, device_mesh=self.tp_mesh, current_placement=self.placement)

        dtype = x.dtype

        x = x.float()
        x = F.normalize(x, p=self.p, eps=self.eps, dim=-1)
        x = x.to(dtype)

        if self.weight is not None:
            x = self.weight * x

        if self.is_tp_enabled:
            x = dtensor_to_tensor(x, device_mesh=self.tp_mesh, desired_placement=self.placement)

        return x


_NORMALIZATION_FUNCTIONS = {"layernorm": LayerNorm, "p_norm": PNorm, "rmsnorm": RMSNorm}


def get_normalization_function(
    normalization_function: str,
    normalized_shape: int,
    eps: float = 1e-5,
    p: int | None = None,
    use_padding_free_transformer: bool = False,
    sequence_parallel: bool = False,
) -> LayerNorm | RMSNorm | PNorm:
    if normalization_function is None:
        return nn.Identity()

    kwargs = {
        "normalized_shape": normalized_shape,
        "eps": eps,
        "use_padding_free_transformer": use_padding_free_transformer,
        "sequence_parallel": sequence_parallel,
    }

    if normalization_function in _NORMALIZATION_FUNCTIONS:
        if normalization_function == "p_norm":
            assert p is not None
            normalization = _NORMALIZATION_FUNCTIONS[normalization_function](**kwargs, p=p)
        else:
            assert p is None
            normalization = _NORMALIZATION_FUNCTIONS[normalization_function](**kwargs)
    else:
        raise ValueError(f"unexpected `normalization_function` {normalization_function}")

    for parameter in normalization.parameters():
        mark_parameter_as_no_weight_decay(parameter)

    return normalization
