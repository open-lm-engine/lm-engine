# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any

import torch

from .packages import is_torch_neuronx_available, is_torch_xla_available


if is_torch_xla_available():
    from torch_xla.core.xla_model import get_rng_state as xla_get_rng_state
    from torch_xla.core.xla_model import set_rng_state as xla_set_rng_state
    from torch_xla.core.xla_model import xla_device


_IS_ROCM_AVAILABLE = torch.version.hip is not None


class Accelerator(Enum):
    cpu = "cpu"
    cuda = "cuda"
    rocm = "rocm"
    tpu = "tpu"
    trainium = "trainium"

    @staticmethod
    @lru_cache
    def get_accelerator() -> Accelerator:
        if torch.cuda.is_available():
            accelerator = Accelerator.rocm if _IS_ROCM_AVAILABLE else Accelerator.cuda
        elif is_torch_xla_available():
            accelerator = Accelerator.tpu
        elif is_torch_neuronx_available():
            accelerator = Accelerator.trainium
        else:
            accelerator = Accelerator.cpu

        return accelerator

    @staticmethod
    def get_current_device() -> int | str:
        accelerator = Accelerator.get_accelerator()

        if accelerator in [Accelerator.cuda, Accelerator.rocm]:
            device = torch.cuda.current_device()
        elif accelerator == Accelerator.tpu:
            device = xla_device()
        elif accelerator == Accelerator.trainium:
            device = torch.neuron.current_device()
        elif accelerator == Accelerator.cpu:
            device = "cpu"

        return device

    @staticmethod
    @lru_cache
    def get_device_type() -> str:
        accelerator = Accelerator.get_accelerator()

        if accelerator in [Accelerator.cuda, Accelerator.rocm]:
            device = "cuda"
        elif accelerator == Accelerator.tpu:
            device = "xla"
        elif accelerator == Accelerator.trainium:
            device = "neuron"
        elif accelerator == Accelerator.cpu:
            device = "cpu"

        return device

    @staticmethod
    def set_device(device: int) -> None:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.cuda:
            torch.cuda.set_device(device)
        elif accelerator == Accelerator.trainium:
            torch.neuron.set_device(device)

    @staticmethod
    def get_rng_state() -> Any:
        accelerator = Accelerator.get_accelerator()

        if accelerator in [Accelerator.cuda, Accelerator.rocm]:
            state = torch.cuda.get_rng_state()
        elif accelerator == Accelerator.tpu:
            state = xla_get_rng_state()
        elif accelerator == Accelerator.trainium:
            state = torch.neuron.get_rng_state()
        else:
            raise ValueError(f"unexpected device ({accelerator})")

        return state

    @staticmethod
    def set_rng_state(state: Any) -> Any:
        accelerator = Accelerator.get_accelerator()

        if accelerator in [Accelerator.cuda, Accelerator.rocm]:
            state = torch.cuda.set_rng_state(state)
        elif accelerator == Accelerator.tpu:
            state = xla_set_rng_state(state)
        elif accelerator == Accelerator.trainium:
            state = torch.neuron.set_rng_state(state)
        else:
            raise ValueError(f"unexpected device ({accelerator})")

        return state
