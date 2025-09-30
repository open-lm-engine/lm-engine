# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from dataclasses import dataclass

import torch
from fma import pack_sequence, unpack_sequence


@dataclass
class PackedTensor:
    tensor: torch.Tensor
    assume_ragged: bool
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None
    batch_size: int | None = None

    @staticmethod
    def from_torch_tensor(
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        is_packed: bool = False,
    ) -> PackedTensor:
        assert batch_size is not None or cu_seqlens is not None
        assume_ragged = cu_seqlens is not None

        if assume_ragged:
            assert max_seqlen is not None

            if batch_size is None:
                batch_size = cu_seqlens.size(0) - 1

            assert cu_seqlens.size(0) - 1 == batch_size

            packed_tensor = pack_sequence(inputs=tensor, cu_seqlens=cu_seqlens) if is_packed else tensor
        else:
            assert tensor.size(0) == batch_size

            if max_seqlen is None:
                max_seqlen = tensor.size(1)

            assert tensor.size(1) == max_seqlen

            packed_tensor = tensor

        packed_tensor = PackedTensor(
            tensor=packed_tensor,
            assume_ragged=assume_ragged,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
        )

        return packed_tensor

    def to_torch_tensor(self, output_shape: tuple[int]) -> torch.Tensor:
        if self.assume_ragged:
            tensor = unpack_sequence(inputs=self.tensor, cu_seqlens=self.cu_seqlens, output_shape=output_shape)
        else:
            tensor = self.tensor

        return tensor

    def get_num_tokens(self) -> int:
        T = self.tensor.size(0)
        if not self.assume_ragged:
            T *= self.tensor.size(1)

        return T

    def with_new_data(self, tensor: torch.Tensor) -> PackedTensor:
        return PackedTensor(
            tensor=tensor,
            assume_ragged=self.assume_ragged,
            cu_seqlens=self.cu_seqlens,
            max_seqlen=self.max_seqlen,
            batch_size=self.batch_size,
        )

    def get_last_element_along_sequence(
        self, tensor: torch.Tensor | None = None, cu_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        if tensor is None:
            assert cu_seqlens is None

            tensor = self.tensor
            cu_seqlens = self.cu_seqlens
        else:
            assert cu_seqlens is not None

        if self.assume_ragged:
            tensor = tensor[cu_seqlens[1:] - 1]
        else:
            tensor = tensor[:, -1]

        return tensor

    def get_max_seqlen(self, return_none_allowed: bool = True) -> int:
        if return_none_allowed and not self.assume_ragged:
            return None

        return self.max_seqlen

    def get_cu_seqlens(self, return_none_allowed: bool = True) -> torch.Tensor:
        if return_none_allowed and not self.assume_ragged:
            return None

        if self.cu_seqlens is None:
            self.cu_seqlens = torch.arange(
                0, self.batch_size * self.max_seqlen + 1, self.max_seqlen, device=self.tensor.device
            )

        return self.cu_seqlens
