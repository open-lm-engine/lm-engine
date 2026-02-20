# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....kernels import Kernel, is_kernel_allowed, wait_for_ACT
from ....utils import Accelerator, is_xma_available
from ..chunk import contiguous_chunk
from .base import get_base_activation


if is_xma_available():
    from xma import swiglu_packed


_GLU_BASE_MAPPING = {
    "glu": "sigmoid",
    "sigmoid_glu": "sigmoid",
    "ceglu": "celu",
    "eglu": "elu",
    "geglu": "gelu",
    "miglu": "mish",
    "mishglu": "mish",
    "preglu": "prelu",
    "reglu": "relu",
    "rreglu": "rrelu",
    "seglu": "selu",
    "swiglu": "swish",
}


class GLUActivation(nn.Module):
    def __init__(self, base_activation: nn.Module) -> GLUActivation:
        super().__init__()
        self.base_activation = base_activation

    def forward(self, x: torch.Tensor, is_interleaved: bool) -> torch.Tensor:
        if (
            is_kernel_allowed(Kernel.swiglu_packed)
            and isinstance(self.base_activation, nn.SiLU)
            and not is_interleaved
        ):
            x = wait_for_ACT(x, wait_in_forward=True, wait_in_backward=False)
            x = swiglu_packed(x)
            x = wait_for_ACT(x, wait_in_forward=False, wait_in_backward=True)
        else:
            if is_interleaved:
                u = x[..., 1::2]
                g = x[..., ::2]
            else:
                u, g = (contiguous_chunk if Accelerator.get_accelerator() == Accelerator.trainium else torch.chunk)(
                    x, 2, dim=-1
                )

            x = u * self.base_activation(g)

        return x


def get_glu_activation(name: str) -> nn.GLU | GLUActivation:
    if name in _GLU_BASE_MAPPING:
        name = _GLU_BASE_MAPPING[name]
    elif name.endswith("_glu"):
        name = name.rstrip("_glu")
    else:
        raise ValueError("invalid activation function")

    base_activation = get_base_activation(name)
    activation_function = GLUActivation(base_activation)

    return activation_function


def is_glu(name: str) -> bool:
    return name.endswith("glu")
