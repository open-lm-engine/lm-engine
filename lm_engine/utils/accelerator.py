# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum

import torch


class Accelerator(Enum):
    cuda = "cuda"
    tpu = "tpu"

    @staticmethod
    def get_accelerator_from_tensor(x: torch.Tensor) -> Accelerator:
        device = x.device.type
        return Accelerator.cuda if device == "cuda" else Accelerator.tpu
