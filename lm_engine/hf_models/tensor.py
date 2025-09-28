# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from fma import pack_sequence, unpack_sequence


class PackedTensor(torch.Tensor):
    def __new__(
        cls,
        packed_tensor: torch.Tensor,
        original_shape: tuple[int],
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
    ) -> PackedTensor:
        self = torch.as_tensor(packed_tensor).as_subclass(cls)

        self._packed_tensor = packed_tensor
        self._original_shape = original_shape
        self._cu_seqlens = cu_seqlens
        self._max_seqlen = max_seqlen
        self._batch_size = cu_seqlens.size(0) - 1 if batch_size is None else batch_size

        return self

    @staticmethod
    def from_unpacked_tensor(
        unpacked_tensor: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
    ) -> PackedTensor:
        assert batch_size is not None or cu_seqlens is not None

        if batch_size is None:
            batch_size = cu_seqlens.size(0) - 1
            packed_tensor = pack_sequence(inputs=unpacked_tensor, cu_seqlens=cu_seqlens)
        else:
            assert unpacked_tensor.size(0) % batch_size == 0
            packed_tensor = unpacked_tensor.flatten(0, 1)

        packed_tensor = PackedTensor(
            packed_tensor=packed_tensor,
            original_shape=unpacked_tensor.size(),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
        )

        return packed_tensor

    def to_unpacked_tensor(self) -> torch.Tensor:
        if self._cu_seqlens is None:
            unpacked_tensor = self.view(self._batch_size, -1, *self.size()[1:])
        else:
            unpacked_tensor = unpack_sequence(
                inputs=self._packed_tensor, cu_seqlens=self._cu_seqlens, output_shape=self._original_shape
            )

        return unpacked_tensor
