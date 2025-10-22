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
    cpu = "cpu"
    cuda = "cuda"
    tpu = "tpu"

    @staticmethod
    def get_accelerator_from_tensor(x: torch.Tensor) -> Accelerator:
        device = x.device.type

        if device == "cpu":
            accelerator = Accelerator.cpu
        elif device == "cuda":
            accelerator = Accelerator.cuda
        elif device == "xla":
            accelerator = Accelerator.tpu
        else:
            raise ValueError(f"unexpected device ({device})")

        return accelerator

    @staticmethod
    def get_accelerator() -> Accelerator:
        if torch.cuda.is_available():
            accelerator = Accelerator.cuda
        elif is_torch_xla_available():
            accelerator = Accelerator.tpu
        else:
            accelerator = Accelerator.cpu

        return accelerator

    @staticmethod
    def get_current_device() -> int:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.cuda:
            device = torch.cuda.current_device()
        elif accelerator == Accelerator.tpu:
            device = xla_device()
        elif accelerator == Accelerator.cpu:
            device = "cpu"

        return device

    @staticmethod
    def set_device(device: int) -> None:
        if Accelerator.get_accelerator() == Accelerator.cuda:
            torch.cuda.set_device(device)
