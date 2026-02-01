# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLUSquared(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)

        y = x.to(torch.float32)
        y *= y
        y = y.type_as(x)

        return y


_BASE_ACTIVATIONS = {
    "celu": nn.CELU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "gelu_pytorch_tanh": (nn.GELU, {"approximate": "tanh"}),
    "selu": nn.SELU,
    "hard_shrink": nn.Hardshrink,
    "hard_sigmoid": nn.Hardsigmoid,
    "hard_swish": nn.Hardswish,
    "hard_tanh": nn.Hardtanh,
    "identity": nn.Identity,
    "leaky_reLU": nn.LeakyReLU,
    "log_sigmoid": nn.LogSigmoid,
    "mish": nn.Mish,
    "prelu": nn.PReLU,
    "relu": nn.ReLU,
    "relu2": ReLUSquared,
    "relu_squared": ReLUSquared,
    "relu6": nn.ReLU6,
    "rrelu": nn.RReLU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "softplus": nn.Softplus,
    "soft_plus": nn.Softplus,
    "soft_shrink": nn.Softshrink,
    "soft_sign": nn.Softsign,
    "tanh": nn.Tanh,
    "tanh_shrink": nn.Tanhshrink,
}


def get_base_activation(name: str) -> nn.Module:
    if name in _BASE_ACTIVATIONS:
        activation = _BASE_ACTIVATIONS[name]

        if isinstance(activation, tuple):
            activation = activation[0](**activation[1])
        else:
            activation = activation()

        return activation

    raise ValueError("invalid activation function")
