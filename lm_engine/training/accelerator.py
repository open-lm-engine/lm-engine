# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import random
from enum import Enum
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch.profiler import ProfilerActivity

from .utils import is_torch_neuronx_available, is_torch_xla_available


if is_torch_xla_available():
    from torch_xla.core.xla_model import get_rng_state as xla_get_rng_state
    from torch_xla.core.xla_model import mark_step as xla_mark_step
    from torch_xla.core.xla_model import set_rng_state as xla_set_rng_state
    from torch_xla.core.xla_model import xla_device


_IS_ROCM_AVAILABLE = torch.version.hip is not None


class KernelBackend(Enum):
    cuda = "cuda"
    jax = "jax"
    mps = "mps"
    nki = "nki"
    pallas = "pallas"
    rocm = "rocm"
    torch = "torch"
    triton = "triton"

    def get_compatible_accelerator(self) -> Accelerator:
        found_accelerator = Accelerator.get_accelerator()

        if self == KernelBackend.torch or (
            self == KernelBackend.triton and found_accelerator in [Accelerator.cuda, Accelerator.rocm]
        ):
            return found_accelerator

        mapping = {
            KernelBackend.cuda: Accelerator.cuda,
            KernelBackend.mps: Accelerator.mps,
            KernelBackend.nki: Accelerator.trainium,
            KernelBackend.pallas: Accelerator.tpu,
            KernelBackend.rocm: Accelerator.rocm,
        }

        return mapping.get(self, None)

    def verify_accelerator(self) -> bool:
        expected_accelerator = self.get_compatible_accelerator()
        found_accelerator = Accelerator.get_accelerator()
        return expected_accelerator == found_accelerator


class Accelerator(Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"
    rocm = "rocm"
    tpu = "tpu"
    trainium = "trainium"

    @staticmethod
    @lru_cache
    def get_accelerator() -> Accelerator:
        if torch.cuda.is_available():
            accelerator = Accelerator.rocm if _IS_ROCM_AVAILABLE else Accelerator.cuda
        elif torch.mps.is_available():
            accelerator = Accelerator.mps
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
        elif accelerator == Accelerator.mps:
            device = "mps"
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
        elif accelerator == Accelerator.mps:
            device = "mps"
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
        elif accelerator == Accelerator.mps:
            state = torch.mps.get_rng_state()
        elif accelerator == Accelerator.tpu:
            state = xla_get_rng_state()
        elif accelerator == Accelerator.trainium:
            state = torch.neuron.get_rng_state()
        elif accelerator == Accelerator.cpu:
            state = torch.get_rng_state()
        else:
            raise ValueError(f"unexpected device ({accelerator})")

        return state

    @staticmethod
    def set_rng_state(state: Any) -> Any:
        accelerator = Accelerator.get_accelerator()

        if accelerator in [Accelerator.cuda, Accelerator.rocm]:
            state = torch.cuda.set_rng_state(state)
        elif accelerator == Accelerator.mps:
            state = torch.mps.set_rng_state(state)
        elif accelerator == Accelerator.tpu:
            state = xla_set_rng_state(state)
        elif accelerator == Accelerator.trainium:
            state = torch.neuron.set_rng_state(state)
        elif accelerator == Accelerator.cpu:
            state = torch.set_rng_state(state)
        else:
            raise ValueError(f"unexpected device ({accelerator})")

        return state

    @staticmethod
    @lru_cache
    def get_kernel_backend() -> KernelBackend:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.cuda:
            kernel_backend = KernelBackend.rocm if _IS_ROCM_AVAILABLE else KernelBackend.cuda
        elif accelerator == Accelerator.mps:
            kernel_backend = KernelBackend.mps
        elif accelerator == Accelerator.tpu:
            kernel_backend = KernelBackend.pallas
        elif accelerator == Accelerator.trainium:
            kernel_backend = KernelBackend.nki
        else:
            kernel_backend = KernelBackend.triton

        return kernel_backend

    @staticmethod
    def get_profiler_activity() -> ProfilerActivity:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.trainium:
            return ProfilerActivity.PrivateUse1

        return ProfilerActivity.CUDA

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        accelerator = Accelerator.get_accelerator()

        if accelerator in [Accelerator.cuda, Accelerator.rocm]:
            torch.cuda.manual_seed_all(seed)
        elif accelerator == Accelerator.mps:
            torch.mps.manual_seed(seed)

    @staticmethod
    def get_torch_compile_backend() -> str:
        if Accelerator.get_accelerator() == Accelerator.trainium:
            return "neuron"

        return "inductor"

    @staticmethod
    def synchronize() -> None:
        accelerator = Accelerator.get_accelerator()

        if accelerator in [Accelerator.cuda, Accelerator.rocm]:
            torch.cuda.synchronize()
        elif accelerator == Accelerator.mps:
            torch.mps.synchronize()
        elif accelerator == Accelerator.tpu:
            xla_mark_step()
        elif accelerator == Accelerator.trainium:
            torch.neuron.synchronize()
