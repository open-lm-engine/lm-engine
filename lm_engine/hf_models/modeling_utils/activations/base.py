# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch.nn as nn
from transformers.activations import ACT2CLS, ClassInstantier


_BASE_ACTIVATIONS = {
    "celu": nn.modules.CELU,
    "elu": nn.modules.ELU,
    "gelu": nn.modules.GELU,
    "gelu_pytorch_tanh": (nn.modules.GELU, {"approximate": "tanh"}),
    "selu": nn.modules.SELU,
    "hard_shrink": nn.modules.Hardshrink,
    "hard_sigmoid": nn.modules.Hardsigmoid,
    "hard_swish": nn.modules.Hardswish,
    "hard_tanh": nn.modules.Hardtanh,
    "identity": nn.modules.Identity,
    "laplace": ACT2CLS["laplace"],
    "leaky_reLU": nn.modules.LeakyReLU,
    "log_sigmoid": nn.modules.LogSigmoid,
    "mish": nn.modules.Mish,
    "prelu": nn.modules.PReLU,
    "relu": nn.modules.ReLU,
    "relu2": ACT2CLS["relu2"],
    "relu_squared": ACT2CLS["relu2"],
    "relu6": nn.modules.ReLU6,
    "rrelu": nn.modules.RReLU,
    "sigmoid": nn.modules.Sigmoid,
    "silu": nn.modules.SiLU,
    "swish": nn.modules.SiLU,
    "softplus": nn.modules.Softplus,
    "soft_plus": nn.modules.Softplus,
    "soft_shrink": nn.modules.Softshrink,
    "soft_sign": nn.modules.Softsign,
    "tanh": nn.modules.Tanh,
    "tanh_shrink": nn.modules.Tanhshrink,
}
# instantiates the module when __getitem__ is called
_BASE_ACTIVATIONS = ClassInstantier(_BASE_ACTIVATIONS)


def get_base_activation(name: str) -> nn.Module:
    if name in _BASE_ACTIVATIONS:
        return _BASE_ACTIVATIONS[name]
    raise ValueError("invalid activation function")
