# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import random

import numpy as np
import torch

from .accelerator import Accelerator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    accelerator = Accelerator.get_accelerator()

    if accelerator == Accelerator.cuda:
        torch.cuda.manual_seed_all(seed)
    elif accelerator == Accelerator.mps:
        torch.mps.manual_seed(seed)
