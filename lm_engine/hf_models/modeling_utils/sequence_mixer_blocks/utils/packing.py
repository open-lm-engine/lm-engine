# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

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

    inputs = _pack_sequence(inputs=inputs, cu_seqlens=cu_seqlens, total_tokens=cu_seqlens[-1].item())

    if is_tensor:
        inputs = inputs[0]

    return inputs


def unpack_sequence(
    inputs: torch.Tensor | list[torch.Tensor], cu_seqlens: torch.Tensor, output_shape: tuple[int]
) -> torch.Tensor | list[torch.Tensor]:
    is_tensor = isinstance(inputs, torch.Tensor)
    if is_tensor:
        inputs = [inputs]

    inputs = _unpack_sequence(
        inputs=inputs, cu_seqlens=cu_seqlens, batch_size=output_shape[0], sequence_length=output_shape[1]
    )

    if is_tensor:
        inputs = inputs[0]

    return inputs
