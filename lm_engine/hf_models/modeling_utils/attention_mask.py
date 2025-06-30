# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import ConfigDict

from ...enums import Kernel
from ...kernels import is_kernel_allowed
from ...utils import BaseArgs, is_cute_kernels_available
from ..cache import GenerationCache


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
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), arbitrary_types_allowed=True)

    attention_mask: torch.Tensor | None = None
    linear_mask: torch.Tensor | None = None
    causal_mask: torch.Tensor | None = None

    cu_seqlens: torch.Tensor | None = None
    batch_size: int | None = None
    max_seqlen: int | None = None
    total_tokens: int | None = None
    all_sequences_of_equal_length: bool

    _mask_value: torch.Tensor | None = None

    @staticmethod
    @torch.no_grad()
    def from_metadata(batch_size: int, sequence_length: int) -> AttentionMaskInfo:
        return AttentionMaskInfo(
            max_seqlen=sequence_length, total_tokens=batch_size * sequence_length, all_sequences_of_equal_length=True
        )

    @staticmethod
    @torch.no_grad()
    def from_attention_mask(attention_mask: torch.Tensor | None = None) -> AttentionMaskInfo:
        assert attention_mask.dim() == 2

        seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        B, S = attention_mask.size()

        return AttentionMaskInfo(
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            batch_size=B,
            max_seqlen=S,
            total_tokens=B * S,
            all_sequences_of_equal_length=False,
        )

    @staticmethod
    @torch.no_grad()
    def from_cu_seqlens(cu_seqlens: torch.Tensor, all_sequences_of_equal_length: bool) -> AttentionMaskInfo:
        B = cu_seqlens.size(0) - 1
        seqlen = cu_seqlens[1:] - cu_seqlens[:-1]
        # we move to CPU here otherwise FlashAttention will move to CPU on every invocation i.e all layers
        max_seqlen = seqlen.max().item()

        if all_sequences_of_equal_length:
            attention_mask_info = AttentionMaskInfo(
                cu_seqlens=cu_seqlens,
                batch_size=B,
                max_seqlen=max_seqlen,
                all_sequences_of_equal_length=all_sequences_of_equal_length,
            )
        else:
            attention_mask_info = AttentionMaskInfo(
                cu_seqlens=cu_seqlens,
                batch_size=B,
                max_seqlen=max_seqlen,
                total_tokens=cu_seqlens[-1].item(),
                all_sequences_of_equal_length=all_sequences_of_equal_length,
            )

        return attention_mask_info

    @torch.no_grad()
    def get_attention_mask(self, return_none_allowed: bool) -> torch.Tensor | None:
        if return_none_allowed and self.all_sequences_of_equal_length:
            return None

        if self.attention_mask is None:
            if self.cu_seqlens is None:
                assert self.batch_size is not None
                assert self.max_seqlen is not None

                self.attention_mask = torch.ones(
                    self.batch_size, self.max_seqlen, dtype=torch.int32, device=torch.cuda.current_device()
                )
            else:
                self.attention_mask = unpack_sequence(
                    inputs=torch.ones((self.total_tokens,), dtype=torch.int32, device=self.cu_seqlens.device),
                    cu_seqlens=self.cu_seqlens,
                    desired_shape=(self.batch_size, self.max_seqlen),
                )

        return self.attention_mask

    @torch.no_grad()
    def get_linear_mask(self, cache: GenerationCache) -> torch.Tensor | None:
        if self.linear_mask is None:
            attention_mask = self.get_attention_mask()
            linear_mask = attention_mask

            if (
                cache is None
                or cache.get_seq_length() > 0
                or (attention_mask is not None and torch.all(attention_mask == 1))
            ):
                linear_mask = None

            self.linear_mask = linear_mask

        return self.linear_mask

    @torch.no_grad()
    def get_causal_mask(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        attention_mask = self.get_attention_mask()

        if attention_mask is not None:
            if self._mask_value is None or self._mask_value.dtype != dtype or self._mask_value.device != device:
                self._mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)

            # we use the causal/non-causal argument of SDPA for attention in this case
            past_length = key_length - query_length

            if query_length > 1:
                causal_mask = torch.empty((query_length, key_length), dtype=torch.bool, device=device)
                causal_mask[:, past_length:] = torch.tril(
                    torch.ones(query_length, query_length, dtype=torch.bool, device=device)
                )

                if past_length > 0:
                    causal_mask[:, :past_length] = True

                causal_mask = causal_mask.unsqueeze(0)

                if attention_mask is None:
                    causal_mask = causal_mask.expand(batch_size, -1, -1)
                else:
                    causal_mask = causal_mask & attention_mask.unsqueeze(1).to(torch.bool)
            else:
                if attention_mask is None:
                    causal_mask = torch.ones(batch_size, query_length, key_length, dtype=torch.bool, device=device)
                else:
                    causal_mask = attention_mask.unsqueeze(1).to(dtype=torch.bool, device=device)

            causal_mask = causal_mask.unsqueeze(1)
            causal_mask = torch.where(attention_mask, ~attention_mask, self._mask_value)
            causal_mask = causal_mask * ~torch.all(attention_mask == self._mask_value, dim=-1, keepdim=True)

            self.causal_mask = causal_mask

        return self.causal_mask
