# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn

from ....kernels import Kernel, is_kernel_allowed, wait_for_ACT
from ....utils import is_cute_kernels_available
from .base import get_base_activation


if is_cute_kernels_available():
    from cute_kernels import swiglu_packed_cute


_GLU_BASE_MAPPING = {
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
    def __init__(self, base_activation: nn.Module) -> None:
        super().__init__()
        self.base_activation = base_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_kernel_allowed(Kernel.swiglu_packed_cute) and isinstance(self.base_activation, nn.SiLU):
            x = wait_for_ACT(x, wait_in_forward=True, wait_in_backward=False)
            x = swiglu_packed_cute(x)
            x = wait_for_ACT(x, wait_in_forward=False, wait_in_backward=True)
        else:
            x = x.chunk(2, dim=-1)
            x = x[0] * self.base_activation(x[1])

        return x


def get_glu_activation(name: str) -> nn.Module:
    # for glu and sigmoid_glu, we directly return the pytorch's GLU
    if name in ["glu", "sigmoid_glu"]:
        activation_function = nn.GLU()
    else:
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
