# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from dataclasses import dataclass, field

import torch

from ..enums import Kernel
from ..generation_cache import GenerationCache
from ..kernels import is_kernel_allowed
from .position_embedding import PositionInfo


@dataclass
class AttentionMaskInfo:
    attention_mask: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None
    causal_mask: torch.Tensor | None = None
    linear_attention_mask: torch.Tensor | None = None

    _causal_mask_computed: bool = field(default=False, repr=False)
    _linear_attention_mask_computed: bool = field(default=False, repr=False)

    def reset_parameters(
        self, batch_size: int, query_length: int, key_length: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        if self._causal_mask_computed:
            return

        if is_kernel_allowed(Kernel.flash_attention_2) or is_kernel_allowed(Kernel.flash_attention_3):
            # we use the causal/non-causal argument of SDPA for attention in this case
            self.causal_mask = self.attention_mask
        elif self.attention_mask is not None:
            causal_mask = _prepare_causal_attention_mask(
                self.attention_mask, batch_size, query_length, key_length, device
            )

            mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
            causal_mask = torch.where(causal_mask, ~causal_mask, mask_value)

            # this is needed to prevent NaN since SDPA
            # see issue: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = causal_mask * ~torch.all(causal_mask == mask_value, dim=-1, keepdim=True)

            self.causal_mask = causal_mask

        self._causal_mask_computed = True

    def get_linear_attention_mask(self, cache_params: GenerationCache | None) -> torch.Tensor | None:
        if not self._linear_attention_mask_computed:
            linear_attention_mask = self.attention_mask
            if (
                cache_params is None
                or cache_params.get_seq_length() > 0
                or (self.attention_mask is not None and torch.all(self.attention_mask == 1))
            ):
                linear_attention_mask = None

            self.linear_attention_mask = linear_attention_mask
            self._linear_attention_mask_computed = True

        return self.linear_attention_mask


def resolve_attention_and_position_info(
    attention_mask_info: AttentionMaskInfo | None, position_info: PositionInfo | None
) -> tuple[AttentionMaskInfo, PositionInfo]:
    # forward() signatures default these to `None` (not `AttentionMaskInfo()`/`PositionInfo()`) since
    # both classes are mutated in place; a mutable default argument would be constructed once and
    # shared/corrupted across every call that omits it
    if attention_mask_info is None:
        attention_mask_info = AttentionMaskInfo()
    if position_info is None:
        position_info = PositionInfo()

    return attention_mask_info, position_info


def _prepare_causal_attention_mask(
    attention_mask: torch.Tensor | None, batch_size: int, query_length: int, key_length: int, device: torch.device
) -> torch.Tensor:
    past_length = key_length - query_length

    if query_length > 1:
        # (query_length, key_length)
        causal_mask = torch.empty((query_length, key_length), dtype=torch.bool, device=device)
        causal_mask[:, past_length:] = torch.tril(
            torch.ones(query_length, query_length, dtype=torch.bool, device=device)
        )

        if past_length > 0:
            causal_mask[:, :past_length] = True

        # (query_length, key_length) -> (1, query_length, key_length)
        causal_mask = causal_mask.unsqueeze(0)

        if attention_mask is None:
            # (1, query_length, key_length) -> (batch_size, query_length, key_length)
            causal_mask = causal_mask.expand(batch_size, -1, -1)
        else:
            # (1, query_length, key_length) & (batch_size, 1, key_length) -> (batch_size, query_length, key_length)
            causal_mask = causal_mask & attention_mask.unsqueeze(1).to(torch.bool)
    else:
        if attention_mask is None:
            # (batch_size, query_length, key_length)
            causal_mask = torch.ones(batch_size, query_length, key_length, dtype=torch.bool, device=device)
        else:
            # (batch_size, query_length, key_length)
            causal_mask = attention_mask.unsqueeze(1).to(dtype=torch.bool, device=device)

    causal_mask = causal_mask.unsqueeze(1)

    return causal_mask
