# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...kernels import KernelBackend
from ...kernels.functional import pack_sequence as _pack_sequence
from ...kernels.functional import unpack_sequence as _unpack_sequence
from ..enums import Kernel
from ..kernels import is_kernel_allowed


def compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = seqlens.max().item()
    return cu_seqlens, max_seqlen


def pack_sequence(
    inputs: torch.Tensor | list[torch.Tensor], cu_seqlens: torch.Tensor
) -> torch.Tensor | list[torch.Tensor]:
    is_tensor = isinstance(inputs, torch.Tensor)
    if is_tensor:
        inputs = [inputs]

    outputs = _pack_sequence(
        inputs=inputs,
        cu_seqlens=cu_seqlens,
        total_tokens=cu_seqlens[-1].item(),
        kernel_backend=None if is_kernel_allowed(Kernel.pack_sequence) else KernelBackend.torch,
    )

    if is_tensor:
        outputs = outputs[0]

    return outputs


def unpack_sequence(
    inputs: torch.Tensor | list[torch.Tensor], cu_seqlens: torch.Tensor, output_shape: tuple[int]
) -> torch.Tensor | list[torch.Tensor]:
    is_tensor = isinstance(inputs, torch.Tensor)
    if is_tensor:
        inputs = [inputs]

    outputs = _unpack_sequence(
        inputs=inputs,
        cu_seqlens=cu_seqlens,
        batch_size=output_shape[0],
        sequence_length=output_shape[1],
        kernel_backend=None if is_kernel_allowed(Kernel.unpack_sequence) else KernelBackend.torch,
    )

    if is_tensor:
        outputs = outputs[0]

    return outputs
