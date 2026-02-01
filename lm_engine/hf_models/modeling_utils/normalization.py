# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...enums import Kernel
from ...kernels import is_kernel_allowed
from ...utils import is_xma_available
from ..parameter import mark_parameter_as_initialized, mark_parameter_as_no_weight_decay


if is_xma_available():
    from xma import rmsnorm


class LayerNorm(nn.LayerNorm):
    def reset_parameters(self) -> None:
        super().reset_parameters()
        mark_parameter_as_initialized(self.weight)


class RMSNorm(nn.RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_kernel_allowed(Kernel.rmsnorm) or is_kernel_allowed(Kernel.rmsnorm_memory_efficient):
            hidden_states = rmsnorm(
                x=hidden_states,
                weight=self.weight,
                eps=self.eps,
                memory_efficient=is_kernel_allowed(Kernel.rmsnorm_memory_efficient),
            )
        else:
            hidden_states = super().forward(hidden_states)

        return hidden_states

    def reset_parameters(self) -> None:
        super().reset_parameters()
        mark_parameter_as_initialized(self.weight)


class PNorm(RMSNorm):
    def __init__(
        self,
        normalized_shape: int,
        p: int,
        eps: float | None = None,
        elementwise_affine=True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> PNorm:
        self.p = p
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype

        hidden_states = hidden_states.float()
        hidden_states = F.normalize(hidden_states, p=self.p, eps=self.eps, dim=-1)
        hidden_states = hidden_states.to(dtype)

        if self.weight is not None:
            hidden_states = self.weight * hidden_states

        return hidden_states


_NORMALIZATION_FUNCTIONS = {"layernorm": LayerNorm, "p_norm": PNorm, "rmsnorm": RMSNorm}


def get_normalization_function(
    normalization_function: str, normalized_shape: int, eps: float = 1e-5, p: int | None = None
) -> LayerNorm | RMSNorm | PNorm:
    if normalization_function is None:
        return nn.Identity()

    if normalization_function in _NORMALIZATION_FUNCTIONS:
        if normalization_function == "p_norm":
            assert p is not None
            normalization = _NORMALIZATION_FUNCTIONS[normalization_function](normalized_shape, eps=eps, p=p)
        else:
            assert p is None
            normalization = _NORMALIZATION_FUNCTIONS[normalization_function](normalized_shape, eps=eps)
    else:
        raise ValueError(f"unexpected `normalization_function` {normalization_function}")

    for parameter in normalization.parameters():
        mark_parameter_as_no_weight_decay(parameter)

    return normalization
