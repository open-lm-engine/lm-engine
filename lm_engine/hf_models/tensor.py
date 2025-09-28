# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from contextlib import contextmanager

import torch
from fma import pack_sequence, unpack_sequence


class PackedTensor(torch.Tensor):
    _is_safe = False

    def __new__(
        cls,
        tensor: torch.Tensor,
        original_shape: tuple[int],
        assume_ragged: bool,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
    ) -> PackedTensor:
        self = torch.as_tensor(tensor).as_subclass(cls)

        self._tensor = tensor
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

            packed_tensor = unpacked_tensor

        packed_tensor = PackedTensor(
            packed_tensor=packed_tensor,
            original_shape=unpacked_tensor.size(),
            assume_ragged=assume_ragged,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
        )

        return packed_tensor

    def get_num_tokens(self) -> int:
        T = self.get_raw_data().size(0)
        if not self.is_ragged_tensor():
            T *= self.get_raw_data().size(1)

        return T

    def with_new_data(self, tensor: torch.Tensor) -> PackedTensor:
        return PackedTensor(
            tensor=tensor,
            original_shape=self._original_shape,
            assume_ragged=self._assume_ragged,
            cu_seqlens=self._cu_seqlens,
            max_seqlen=self._max_seqlen,
            batch_size=self._batch_size,
        )

    def to_unpacked_tensor(self) -> torch.Tensor:
        if self.is_ragged_tensor():
            unpacked_tensor = unpack_sequence(
                inputs=self.get_raw_data(), cu_seqlens=self._cu_seqlens, output_shape=self._original_shape
            )
        else:
            unpacked_tensor = self.get_raw_data()

        return unpacked_tensor

    def get_raw_data(self) -> torch.Tensor:
        return self._tensor

    def get_last_element_along_sequence(self) -> torch.Tensor:
        output = self.get_raw_data()

        if self.is_ragged_tensor():
            output = output[self.get_cu_seqlens()[1:] - 1]
        else:
            output = output[:, -1]

        return output

    def is_ragged_tensor(self) -> bool:
        return self._assume_ragged

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_max_seqlen(self, return_none_allowed: bool = True) -> int:
        if return_none_allowed and not self.is_ragged_tensor():
            return None

        return self._max_seqlen

    def get_cu_seqlens(self, return_none_allowed: bool = True) -> torch.Tensor:
        if return_none_allowed and not self.is_ragged_tensor():
            return None

        if self._cu_seqlens is None:
            self._cu_seqlens = torch.arange(
                0, self._batch_size * self._max_seqlen + 1, self._max_seqlen, device=self.device
            )

        return self._cu_seqlens

    @contextmanager
    @classmethod
    def safe_mode(cls):
        cls._is_safe = True
        yield
        cls._is_safe = False

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if cls._is_safe:
            return super().__torch_function__(func, types, args, kwargs)

        raise NotImplementedError("unpack the tensor to run ops on it")
