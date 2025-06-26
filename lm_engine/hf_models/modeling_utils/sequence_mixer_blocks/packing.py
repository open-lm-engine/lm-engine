# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import is_cute_kernels_available


if is_cute_kernels_available():
    from cute_kernels import KernelBackend, pack_sequence_cute, unpack_sequence_cute


def compute_cu_seqlens_and_max_seqlen_from_attention_mask(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = seqlens.max().item()
    return cu_seqlens, max_seqlen


def pack_sequence(
    inputs: torch.Tensor | list[torch.Tensor], cu_seqlens: torch.Tensor
) -> torch.Tensor | list[torch.Tensor]:
    if is_kernel_allowed(Kernel.pack_sequence_cute):
        inputs = pack_sequence_cute(inputs=inputs, cu_seqlens=cu_seqlens)
    else:
        inputs = pack_sequence_cute(
            inputs=inputs,
            cu_seqlens=cu_seqlens,
            kernel_backend_forward=KernelBackend.torch,
            kernel_backend_backward=KernelBackend.torch,
        )

    return inputs


def unpack_sequence(
    inputs: torch.Tensor | list[torch.Tensor], cu_seqlens: torch.Tensor, desired_shape: tuple[int]
) -> torch.Tensor | list[torch.Tensor]:
    if is_kernel_allowed(Kernel.pack_sequence_cute):
        inputs = unpack_sequence_cute(inputs=inputs, cu_seqlens=cu_seqlens, desired_shape=desired_shape)
    else:
        inputs = unpack_sequence_cute(
            inputs=inputs,
            cu_seqlens=cu_seqlens,
            desired_shape=desired_shape,
            kernel_backend_forward=KernelBackend.torch,
            kernel_backend_backward=KernelBackend.torch,
        )

    return inputs
