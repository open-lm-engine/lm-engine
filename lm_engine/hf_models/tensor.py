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
        assume_ragged: bool,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
    ) -> PackedTensor:
        self = torch.as_tensor(tensor).as_subclass(cls)

        self._assume_ragged = assume_ragged
        self._cu_seqlens = cu_seqlens
        self._max_seqlen = max_seqlen
        self._batch_size = batch_size

        return self

    def __init__(
        self,
        tensor: torch.Tensor,
        assume_ragged: bool,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
    ) -> PackedTensor:
        self._tensor = tensor
        self._assume_ragged = assume_ragged
        self._cu_seqlens = cu_seqlens
        self._max_seqlen = max_seqlen
        self._batch_size = batch_size

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
        if self.is_ragged_tensor():
            tensor = unpack_sequence(
                inputs=self.get_raw_data(), cu_seqlens=self._cu_seqlens, output_shape=output_shape
            )
        else:
            tensor = self.get_raw_data()

        return tensor

    def get_num_tokens(self) -> int:
        T = self.get_raw_data().size(0)
        if not self.is_ragged_tensor():
            T *= self.get_raw_data().size(1)

        return T

    def with_new_data(self, tensor: torch.Tensor) -> PackedTensor:
        return PackedTensor(
            tensor=tensor,
            assume_ragged=self._assume_ragged,
            cu_seqlens=self._cu_seqlens,
            max_seqlen=self._max_seqlen,
            batch_size=self._batch_size,
        )

    def get_raw_data(self) -> torch.Tensor:
        return self.as_subclass(torch.Tensor)

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

    def get_dtype(self) -> torch.dtype:
        with self.safe_mode():
            return self.dtype

    def get_device(self) -> torch.device:
        with self.safe_mode():
            return super().get_device()

    @classmethod
    @contextmanager
    def safe_mode(cls):
        cls._is_safe = True
        yield
        cls._is_safe = False

    @classmethod
    def set_safe_mode(cls, enable: bool = False) -> None:
        cls._is_safe = enable

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if cls._is_safe:
            output = super().__torch_dispatch__(func, types, args, kwargs)

            if isinstance(output, PackedTensor):
                return output
            else:
                arg_packed = None

                for arg in args:
                    if isinstance(arg, PackedTensor):
                        arg_packed = arg
                        break

                if arg_packed is None:
                    for arg in kwargs.values():
                        if isinstance(arg, PackedTensor):
                            arg_packed = arg
                            break

                if isinstance(output, torch.Tensor):
                    return arg_packed.with_new_data(output)
                elif isinstance(output, (tuple, list)):
                    output_packed = []
                    for i in output:
                        if isinstance(i, PackedTensor):
                            output_packed.append(i)
                        elif isinstance(i, torch.Tensor):
                            i = arg_packed.with_new_data(i)
                            output_packed.append(i)
                        else:
                            output_packed.append(i)

                    return output_packed
                else:
                    raise ValueError("unexpected output type")

        raise NotImplementedError("unpack the tensor to run ops on it")
