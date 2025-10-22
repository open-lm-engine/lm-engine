# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum

import torch

from .packages import is_torch_xla_available


if is_torch_xla_available():
    from torch_xla.core.xla_model import xla_device


class Accelerator(Enum):
    cuda = "cuda"
    tpu = "tpu"

    @staticmethod
    def get_accelerator_from_tensor(x: torch.Tensor) -> Accelerator:
        device = x.device.type
        return Accelerator.cuda if device == "cuda" else Accelerator.tpu

    @staticmethod
    def get_accelerator() -> Accelerator:
        return Accelerator.cuda if torch.cuda.is_available() else Accelerator.tpu

    @staticmethod
    def get_current_device() -> int:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.cuda:
            device = torch.cuda.current_device()
        elif accelerator == Accelerator.tpu:
            device = xla_device()

        return device

    @staticmethod
    def set_device(device: int) -> None:
        if Accelerator.get_accelerator() == Accelerator.cuda:
            torch.cuda.set_device(device)
