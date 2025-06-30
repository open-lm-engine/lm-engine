# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...enums import Kernel
from ...kernels import is_kernel_allowed
from ...utils import BaseArgs, is_cute_kernels_available


if is_cute_kernels_available():
    from cute_kernels import pack_sequence_cute, unpack_sequence_cute


def pack_sequence(
    inputs: torch.Tensor | list[torch.Tensor], cu_seqlens: torch.Tensor
) -> torch.Tensor | list[torch.Tensor]:
    if is_kernel_allowed(Kernel.pack_sequence_cute):
        outputs = pack_sequence_cute(inputs=inputs, cu_seqlens=cu_seqlens)
    else:
        is_list = isinstance(inputs, (list, tuple))
        if not is_list:
            inputs = [inputs]

        outputs = []

        for x in inputs:
            B, S = x.size()[:2]
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            batch_indices = torch.arange(B, device=x.device).repeat_interleave(seqlens)

            pad_tokens = S - seqlens
            seq_indices = torch.cat([torch.arange(sl, S, device=x.device) for sl in pad_tokens])

            x = x[batch_indices, seq_indices]

            outputs.append(x)

        if not is_list:
            outputs = outputs[0]

    return outputs


def unpack_sequence(
    inputs: torch.Tensor | list[torch.Tensor], cu_seqlens: torch.Tensor, desired_shape: tuple[int]
) -> torch.Tensor | list[torch.Tensor]:
    if is_kernel_allowed(Kernel.unpack_sequence_cute):
        outputs = unpack_sequence_cute(inputs=inputs, cu_seqlens=cu_seqlens, desired_shape=desired_shape)
    else:
        is_list = isinstance(inputs, (list, tuple))
        if not is_list:
            inputs = [inputs]

        outputs = []

        for x in inputs:
            B, S = desired_shape[:2]
            assert cu_seqlens.size(0) - 1 == B
            assert desired_shape[2:] == x.size()[1:]

            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            batch_indices = torch.arange(B, device=x.device).repeat_interleave(seqlens)

            pad_tokens = S - seqlens
            seq_indices = torch.cat([torch.arange(sl, S, device=x.device) for sl in pad_tokens])

            padded = torch.zeros(desired_shape, dtype=x.dtype, device=x.device)
            padded[batch_indices, seq_indices] = x

            outputs.append(padded)

        if not is_list:
            outputs = outputs[0]

    return outputs


class AttentionMaskInfo(BaseArgs):
    attention_mask: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None
    batch_size: int | None = None
    sequence_length: int | None = None
    total_tokens: int | None = None
    all_sequences_of_equal_length: bool

    @staticmethod
    def from_attention_mask(attention_mask: torch.Tensor) -> AttentionMaskInfo:
        seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = seqlens.max().item()

        B, S = attention_mask.size()

        return AttentionMaskInfo(
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=B,
            sequence_length=S,
        )

    @staticmethod
    def from_cu_seqlens(cu_seqlens: torch.Tensor, all_sequences_of_equal_length: bool) -> AttentionMaskInfo:
        B = cu_seqlens.size(0) - 1

        if all_sequences_of_equal_length:
            return AttentionMaskInfo(
                cu_seqlens=cu_seqlens, batch_size=B, all_sequences_of_equal_length=all_sequences_of_equal_length
            )
        else:
            seqlen = cu_seqlens[1:] - cu_seqlens[:-1]
            # we move to CPU here otherwise FlashAttention will move to CPU on every invocation i.e all layers
            max_seqlen = seqlen.max().item()

            return AttentionMaskInfo(
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                batch_size=B,
                total_tokens=cu_seqlens[-1].item(),
                all_sequences_of_equal_length=all_sequences_of_equal_length,
            )

    def get_attention_mask(self) -> torch.Tensor:
        if self.attention_mask is None:
            self.attention_mask = unpack_sequence(
                inputs=torch.ones((self.total_tokens,), dtype=torch.uint32, device=self.cu_seqlens.device),
                cu_seqlens=self.cu_seqlens,
                desired_shape=(self.batch_size, self.sequence_length),
            )

        return self.attention_mask
