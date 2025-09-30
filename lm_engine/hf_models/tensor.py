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
    batch_size: int | None = None

    _assume_ragged: bool | None = None
    _cu_seqlens: torch.Tensor | None = None
    _max_seqlen: int | None = None

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
            batch_size=batch_size,
            _assume_ragged=assume_ragged,
            _cu_seqlens=cu_seqlens,
            _max_seqlen=max_seqlen,
        )

        return packed_tensor

    def to_torch_tensor(self, output_shape: tuple[int]) -> torch.Tensor:
        if self._assume_ragged:
            tensor = unpack_sequence(inputs=self.tensor, cu_seqlens=self._cu_seqlens, output_shape=output_shape)
        else:
            tensor = self.tensor

        return tensor

    def get_num_tokens(self) -> int:
        T = self.tensor.size(0)
        if not self._assume_ragged:
            T *= self.tensor.size(1)

        return T

    def with_new_data(self, tensor: torch.Tensor) -> PackedTensor:
        return PackedTensor(
            tensor=tensor,
            batch_size=self.batch_size,
            _assume_ragged=self._assume_ragged,
            _cu_seqlens=self._cu_seqlens,
            _max_seqlen=self._max_seqlen,
        )

    def get_last_element_along_sequence(
        self, tensor: torch.Tensor | None = None, cu_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        if tensor is None:
            assert cu_seqlens is None

            tensor = self.tensor
            cu_seqlens = self._cu_seqlens
        else:
            assert cu_seqlens is not None

        if self._assume_ragged:
            tensor = tensor[cu_seqlens[1:] - 1]
        else:
            tensor = tensor[:, -1]

        return tensor

    def get_max_seqlen(self, return_none_allowed: bool = True) -> int:
        if return_none_allowed and not self._assume_ragged:
            return None

        return self._max_seqlen

    def get_cu_seqlens(self, return_none_allowed: bool = True) -> torch.Tensor:
        if return_none_allowed and not self._assume_ragged:
            return None

        if self._cu_seqlens is None:
            self._cu_seqlens = torch.arange(
                0, self.batch_size * self._max_seqlen + 1, self._max_seqlen, device=self.tensor.device
            )

        return self._cu_seqlens
