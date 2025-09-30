# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...enums import Kernel
from ...kernels import is_kernel_allowed
from ...utils import is_fma_available


if is_fma_available():
    from fma import KernelBackend
    from fma import pack_sequence as _pack_sequence
    from fma import unpack_sequence as _unpack_sequence


def pack_sequence(
    inputs: torch.Tensor | list[torch.Tensor], cu_seqlens: torch.Tensor
) -> torch.Tensor | list[torch.Tensor]:
    kernel_backend = KernelBackend.cuda if is_kernel_allowed(Kernel.pack_sequence) else KernelBackend.torch

    inputs = _pack_sequence(
        inputs=inputs,
        cu_seqlens=cu_seqlens,
        kernel_backend_forward=kernel_backend,
        kernel_backend_backward=kernel_backend,
    )

    return inputs


def unpack_sequence(
    inputs: torch.Tensor | list[torch.Tensor], cu_seqlens: torch.Tensor, output_shape: tuple[int]
) -> torch.Tensor | list[torch.Tensor]:
    kernel_backend = KernelBackend.cuda if is_kernel_allowed(Kernel.unpack_sequence) else KernelBackend.torch

    inputs = _unpack_sequence(
        inputs=inputs,
        cu_seqlens=cu_seqlens,
        output_shape=output_shape,
        kernel_backend_forward=kernel_backend,
        kernel_backend_backward=kernel_backend,
    )

    return inputs


# NOTE using dataclass here since pydantic doesn't work with torch.compile
@dataclass
class AttentionMaskInfo:
    batch_size: int | None = None
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None
    attention_mask: torch.Tensor | None = None
    causal_mask: torch.Tensor | None = None

    def get_batch_size(self) -> int:
        if self.batch_size is not None:
            return self.batch_size

        if self.cu_seqlens is not None:
            self.batch_size = self.cu_seqlens.size(0) - 1
        elif self.attention_mask is not None:
            self.batch_size = self.attention_mask.size(0)
        else:
            raise NotImplementedError("code is not supposed to reach here")

        return self.batch_size

    def get_cu_seqlens(self, return_none_allowed: bool = True) -> torch.Tensor | None:
        if return_none_allowed:
            return self.cu_seqlens

        if self.cu_seqlens is None:
            seqlens = self.attention_mask.sum(dim=-1, dtype=torch.int32)
            self.cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
            self.max_seqlen = seqlens.max().item()

        return self.cu_seqlens

    def get_max_seqlen(self, return_none_allowed: bool = True) -> int | None:
        if return_none_allowed:
            return self.max_seqlen

        if self.max_seqlen is None:
            # this will cache the max_seqlen
            self.get_cu_seqlens(False)

        return self.max_seqlen

    def get_attention_mask(
        self, return_none_allowed: bool = True, device: torch.device | None = None
    ) -> torch.Tensor | None:
        if return_none_allowed:
            return self.attention_mask

        if self.attention_mask is None:
            cu_seqlens = self.get_cu_seqlens()
            batch_size = self.get_batch_size()
            max_seqlen = self.get_max_seqlen()
            assert max_seqlen is not None

            if cu_seqlens is None:
                self.attention_mask = torch.ones(batch_size, max_seqlen, device=device, dtype=torch.int32)
            else:
                attention_mask_flat = torch.ones_like(cu_seqlens, device=device, dtype=torch.int32)
                self.attention_mask = unpack_sequence(
                    inputs=attention_mask_flat, cu_seqlens=cu_seqlens, output_shape=(batch_size, max_seqlen)
                )

        return self.attention_mask

    def get_causal_mask(
        self, return_none_allowed: bool = True, dtype: torch.dtype | None = None
    ) -> torch.Tensor | None:
        attention_mask = self.get_attention_mask(return_none_allowed)

        if attention_mask is not None:
            _, Q, K = attention_mask.size()
            L = K - Q

            if Q > 1:
                device = attention_mask.device

                causal_mask = torch.empty((Q, K), dtype=torch.bool, device=device)
                causal_mask[:, L:] = torch.tril(torch.ones(Q, K, dtype=torch.bool, device=device))

                if L > 0:
                    causal_mask[:, :L] = True

                causal_mask = causal_mask[None, ...]
                causal_mask = causal_mask & attention_mask[:, None, ...].to(torch.bool)
            elif Q == 1:
                causal_mask = attention_mask[:, None, ...].to(dtype=torch.bool, device=device)
            else:
                raise NotImplementedError("code is not expected to reach here")

            causal_mask = causal_mask[:, None, ...]
            causal_mask = torch.where(causal_mask, ~causal_mask, AttentionMaskInfo._get_mask_value(device, dtype))

            # this is needed to prevent NaN since SDPA
            # see issue: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = causal_mask * ~torch.all(
                causal_mask == AttentionMaskInfo._get_mask_value(device, dtype), dim=-1, keepdim=True
            )

        return attention_mask

    @classmethod
    def _get_mask_value(cls, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if cls.mask_value is None or cls.mask_value.dtype != dtype or cls.mask_value.device != device:
            cls.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        return cls.mask_value
