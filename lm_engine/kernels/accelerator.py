# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from .utils import is_jax_available, is_torch_available, is_torch_neuronx_available


if is_torch_available():
    import torch


_IS_ROCM_AVAILABLE = is_torch_available() and torch.version.hip is not None
_IS_CUDA_AVAILABLE = is_torch_available() and torch.cuda.is_available()
_IS_MPS_AVAILABLE = is_torch_available() and torch.mps.is_available()


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
        is_tpu = False

        if is_jax_available():
            import jax

            is_tpu = jax.default_backend() == "tpu"

        if is_tpu:
            accelerator = Accelerator.tpu
        elif is_torch_neuronx_available():
            accelerator = Accelerator.trainium
        elif _IS_CUDA_AVAILABLE:
            accelerator = Accelerator.rocm if _IS_ROCM_AVAILABLE else Accelerator.cuda
        elif _IS_MPS_AVAILABLE:
            accelerator = Accelerator.mps
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
            raise ValueError("torch does not manage TPU devices; use the JAX API (xma.layers_jax) for TPUs")
        elif accelerator == Accelerator.trainium:
            device = torch.neuron.current_device()
        elif accelerator == Accelerator.cpu:
            device = "cpu"

        return device

    @staticmethod
    def device_count() -> int:
        accelerator = Accelerator.get_accelerator()

        if accelerator in [Accelerator.cuda, Accelerator.rocm]:
            count = torch.cuda.device_count()
        elif accelerator == Accelerator.mps:
            count = 1
        elif accelerator == Accelerator.tpu:
            import jax

            count = jax.device_count()
        elif accelerator == Accelerator.trainium:
            count = torch.neuron.device_count()
        elif accelerator == Accelerator.cpu:
            count = 1

        return count

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
    def synchronize() -> None:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.cuda:
            torch.cuda.synchronize()
        elif accelerator == Accelerator.mps:
            torch.mps.synchronize()

    @staticmethod
    def get_core_count() -> int:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.cuda:
            sm_count = torch.cuda.get_device_properties().multi_processor_count
        else:
            raise ValueError(f"unexpected accelerator ({accelerator})")

        return sm_count

    @staticmethod
    def get_lane_count() -> int:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.tpu:
            lane_count = 128
        else:
            raise ValueError(f"unexpected accelerator ({accelerator})")

        return lane_count
