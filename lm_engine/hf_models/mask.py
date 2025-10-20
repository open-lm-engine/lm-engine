# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..enums import Kernel
from ..kernels import is_kernel_allowed
from ..utils import is_fma_available


if is_fma_available():
    from fma import KernelBackend, pack_sequence, unpack_sequence


_ERROR_MESSAGE = "code is not supposed to reach here"


# NOTE using dataclass here since pydantic doesn't work with torch.compile
@dataclass
class AttentionMaskInfo:
    batch_size: int | None = None
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None
    attention_mask: torch.Tensor | None = None
    device: torch.device | None = None
    mask_value: torch.Tensor | None = None
    causal_mask: torch.Tensor | None = None
    _mask_value: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self._has_cu_seqlens = self.cu_seqlens is not None
        self._has_attention_mask = self.attention_mask is not None

        if self.batch_size is not None:
            assert self.max_seqlen is not None
            assert not self.has_cu_seqlens()
            assert not self.has_attention_mask()
        elif self.cu_seqlens is not None:
            assert self.batch_size is None
            assert self.max_seqlen is not None
            assert not self.has_attention_mask()

            self.device = self.cu_seqlens.device
        elif self.has_attention_mask():
            assert self.batch_size is None
            assert not self.has_cu_seqlens()
            assert self.max_seqlen is None

            self.device = self.attention_mask.device

        assert self.device is not None

    def has_cu_seqlens(self) -> bool:
        return self._has_cu_seqlens

    def has_attention_mask(self) -> bool:
        return self._has_attention_mask

    def has_padding(self) -> bool:
        return self.has_cu_seqlens() or self.has_attention_mask()

    def get_batch_size(self) -> int:
        if self.batch_size is None:
            if self.has_cu_seqlens():
                self.batch_size = self.cu_seqlens.size(0) - 1
            elif self.has_attention_mask():
                self.batch_size = self.attention_mask.size(0)
            else:
                raise NotImplementedError(_ERROR_MESSAGE)

        return self.batch_size

    def get_cu_seqlens(self, return_none_allowed: bool = True) -> torch.Tensor | None:
        if self.has_cu_seqlens():
            return self.cu_seqlens

        if return_none_allowed:
            return None

        if self.has_attention_mask():
            seqlens = self.attention_mask.sum(dim=-1, dtype=torch.int32)
            self.cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
            self.max_seqlen = seqlens.max().item()
        else:
            B = self.get_batch_size()
            S = self.get_max_seqlen()

            self.cu_seqlens = torch.arange(0, B * S + 1, S, dtype=torch.int32, device=self.device)

        return self.cu_seqlens

    def get_max_seqlen(self, return_none_allowed: bool = True) -> int | None:
        if self.has_cu_seqlens():
            assert self.max_seqlen is not None
            return self.max_seqlen

        if return_none_allowed:
            return None

        if self.max_seqlen is None:
            # this will cache the max_seqlen but causes synchronization with CPU
            self.get_cu_seqlens(False)

            if self.max_seqlen is None:
                raise NotImplementedError(_ERROR_MESSAGE)

        return self.max_seqlen

    def get_attention_mask(self, return_none_allowed: bool = True) -> torch.Tensor | None:
        if self.has_attention_mask():
            return self.attention_mask

        if return_none_allowed:
            return None

        B = self.get_batch_size()
        S = self.get_max_seqlen()

        if self.has_cu_seqlens():
            self.attention_mask = self.unpack_sequence(
                inputs=torch.ones_like(self.get_cu_seqlens(), device=self.device, dtype=torch.int32),
                output_shape=(B, S),
            )
        else:
            self.attention_mask = torch.ones(B, S, device=self.device, dtype=torch.int32)

        return self.attention_mask

    def get_position_ids(self) -> torch.Tensor:
        if self.has_cu_seqlens() or self.has_attention_mask():
            attention_mask = self.get_attention_mask(False)
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        else:
            B = self.get_batch_size()
            S = self.get_max_seqlen(False)

            position_ids = torch.arange(0, S, device=self.device)
            position_ids = position_ids[None, ...].expand(B, -1)

        position_ids = self.pack_sequence(position_ids)

        return position_ids

    def get_causal_mask(
        self, query_length: int, return_none_allowed: bool = True, dtype: torch.dtype | None = None
    ) -> torch.Tensor | None:
        if self.causal_mask is not None:
            return self.causal_mask

        if self.has_cu_seqlens() or self.has_attention_mask():
            attention_mask = self.get_attention_mask()
        elif return_none_allowed:
            return None

        Q = query_length
        K = attention_mask.size(1)
        L = K - Q

        if Q > 1:
            causal_mask = torch.empty((Q, K), dtype=torch.bool, device=self.device)
            causal_mask[:, L:] = torch.tril(torch.ones(Q, K, dtype=torch.bool, device=self.device))

            if L > 0:
                causal_mask[:, :L] = True

            causal_mask = causal_mask[None, ...]
            causal_mask = causal_mask & attention_mask[:, None, ...].to(torch.bool)
        elif Q == 1:
            causal_mask = attention_mask[:, None, ...].to(dtype=torch.bool, device=self.device)
        else:
            raise NotImplementedError(_ERROR_MESSAGE)

        causal_mask = causal_mask[:, None, ...]
        causal_mask = torch.where(causal_mask, ~causal_mask, self._get_mask_value(attention_mask.device, dtype))

        # this is needed to prevent NaN since SDPA
        # see issue: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = causal_mask * ~torch.all(
            causal_mask == self._get_mask_value(self.device, dtype), dim=-1, keepdim=True
        )

        self.causal_mask = causal_mask

        return self.causal_mask

    def pack_sequence(self, inputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        if inputs is None:
            return None

        is_tensor = isinstance(inputs, torch.Tensor)
        if is_tensor:
            inputs = [inputs]

        if self.has_cu_seqlens() or self.has_attention_mask():
            kernel_backend = KernelBackend.cuda if is_kernel_allowed(Kernel.pack_sequence) else KernelBackend.torch
            cu_seqlens = self.get_cu_seqlens(False)

            inputs = pack_sequence(
                inputs=inputs,
                cu_seqlens=cu_seqlens,
                total_tokens=cu_seqlens[-1].item(),
                kernel_backend_forward=kernel_backend,
                kernel_backend_backward=kernel_backend,
            )
        else:
            inputs = [i.flatten(0, 1) for i in inputs]

        if is_tensor:
            inputs = inputs[0]

        return inputs

    def unpack_sequence(self, inputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        if inputs is None:
            return None

        is_tensor = isinstance(inputs, torch.Tensor)
        if is_tensor:
            inputs = [inputs]

        B = self.get_batch_size()
        S = self.get_max_seqlen(False)

        if self.has_cu_seqlens() or self.has_attention_mask():
            kernel_backend = KernelBackend.cuda if is_kernel_allowed(Kernel.unpack_sequence) else KernelBackend.torch

            inputs = unpack_sequence(
                inputs=inputs,
                cu_seqlens=self.get_cu_seqlens(False),
                batch_size=B,
                sequence_length=S,
                kernel_backend_forward=kernel_backend,
                kernel_backend_backward=kernel_backend,
            )
        else:
            inputs = [i.reshape(B, S, *i.size()[1:]) for i in inputs]

        if is_tensor:
            inputs = inputs[0]

        return inputs

    def _get_mask_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        return self.mask_value
