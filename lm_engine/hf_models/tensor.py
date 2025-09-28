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
        assume_ragged: bool,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
    ) -> PackedTensor:
        self = torch.as_tensor(packed_tensor).as_subclass(cls)

        self._packed_tensor = packed_tensor
        self._original_shape = original_shape
        self._assume_ragged = assume_ragged
        self._cu_seqlens = cu_seqlens
        self._max_seqlen = max_seqlen
        self._batch_size = batch_size

        return self

    @staticmethod
    def from_unpacked_tensor(
        unpacked_tensor: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
    ) -> PackedTensor:
        assert batch_size is not None or cu_seqlens is not None
        assume_ragged = cu_seqlens is not None

        if assume_ragged:
            assert max_seqlen is not None

            if batch_size is None:
                batch_size = cu_seqlens.size(0) - 1

            assert cu_seqlens.size(0) - 1 == batch_size

            packed_tensor = pack_sequence(inputs=unpacked_tensor, cu_seqlens=cu_seqlens)
        else:
            assert unpacked_tensor.size(0) % batch_size == 0

            if max_seqlen is None:
                max_seqlen = unpacked_tensor.size(1)

            assert unpacked_tensor.size(1) == max_seqlen

            packed_tensor = unpacked_tensor.flatten(0, 1)

        packed_tensor = PackedTensor(
            packed_tensor=packed_tensor,
            original_shape=unpacked_tensor.size(),
            assume_ragged=assume_ragged,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
        )

        return packed_tensor

    def to_unpacked_tensor(self) -> torch.Tensor:
        if self._assume_ragged:
            unpacked_tensor = unpack_sequence(
                inputs=self._packed_tensor, cu_seqlens=self._cu_seqlens, output_shape=self._original_shape
            )
        else:
            unpacked_tensor = self.view(self._batch_size, -1, *self.size()[1:])

        return unpacked_tensor

    def is_ragged_tensor(self) -> bool:
        return self._assume_ragged

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_max_seqlen(self) -> int:
        return self._max_seqlen

    def get_cu_seqlens(self, force_compute: bool = False) -> torch.Tensor:
        if force_compute:
            if self._cu_seqlens is None:
                self._cu_seqlens = torch.arange(
                    0, self._batch_size * self._max_seqlen + 1, self._max_seqlen, device=self.device
                )
        else:
            raise NotImplementedError("code is not supposed to reach here")

        return self._cu_seqlens

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError("unpack the tensor to run ops on it")
