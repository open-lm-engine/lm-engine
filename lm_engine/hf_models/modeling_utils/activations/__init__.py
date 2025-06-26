# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch.nn as nn

from .base import get_base_activation
from .glu import get_glu_activation, is_glu


def get_activation_function(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()

    return get_glu_activation(name) if is_glu(name) else get_base_activation(name)
