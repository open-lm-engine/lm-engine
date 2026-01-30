# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from .....enums import Kernel
from .....kernels import is_kernel_allowed
from .....utils import is_xma_available


if is_xma_available():
    from xma import pack_sequence as _pack_sequence
    from xma import unpack_sequence as _unpack_sequence


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
    is_tensor = isinstance(inputs, torch.Tensor)
    if is_tensor:
        inputs = [inputs]

    if is_kernel_allowed(Kernel.pack_sequence):
        outputs = _pack_sequence(inputs=inputs, cu_seqlens=cu_seqlens, total_tokens=cu_seqlens[-1].item())
    else:
        outputs = []

        for x in inputs:
            assert x.dim() >= 2
            assert x.size(0) == cu_seqlens.size(0) - 1

            B, S = x.size()[:2]
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            batch_indices = torch.arange(B, device=x.device).repeat_interleave(seqlens)

            pad_tokens = S - seqlens
            seq_indices = torch.cat([torch.arange(sl, S, device=x.device) for sl in pad_tokens])

            x = x[batch_indices, seq_indices]

            outputs.append(x)

    if is_tensor:
        outputs = outputs[0]

    return outputs


def unpack_sequence(
    inputs: torch.Tensor | list[torch.Tensor], cu_seqlens: torch.Tensor, output_shape: tuple[int]
) -> torch.Tensor | list[torch.Tensor]:
    is_tensor = isinstance(inputs, torch.Tensor)
    if is_tensor:
        inputs = [inputs]

    if is_kernel_allowed(Kernel.unpack_sequence):
        outputs = _unpack_sequence(
            inputs=inputs, cu_seqlens=cu_seqlens, batch_size=output_shape[0], sequence_length=output_shape[1]
        )
    else:
        outputs = []

        for x in inputs:
            B = output_shape[0]
            S = output_shape[1]

            assert x.dim() >= 2
            assert cu_seqlens.size(0) - 1 == B

            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            batch_indices = torch.arange(B, device=x.device).repeat_interleave(seqlens)

            pad_tokens = S - seqlens
            seq_indices = torch.cat([torch.arange(sl, S, device=x.device) for sl in pad_tokens])

            padded = torch.zeros(output_shape, dtype=x.dtype, device=x.device)
            padded[batch_indices, seq_indices] = x

            outputs.append(x)

    if is_tensor:
        outputs = outputs[0]

    return outputs
