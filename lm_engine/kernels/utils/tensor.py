# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ..math import get_powers_of_2


def get_alignment(x: torch.Tensor) -> int:
    x = x.data_ptr()

    alignment = 4
    for i in get_powers_of_2(4, 16):
        if x % i != 0:
            break
        else:
            alignment = i

    return alignment
