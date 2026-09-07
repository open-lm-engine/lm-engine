# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import random

import numpy as np

from ...utils import is_torch_available


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if is_torch_available():
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
