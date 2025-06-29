# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from .utils import pad


class AlphaNumericTokenizer:
    def __init__(self, lowercase_only: bool = True) -> AlphaNumericTokenizer:
        self.eos_token = "<|endoftext|>"
        self.pad_token = self.eos_token
        self.lowercase_only = lowercase_only

        self.eos_token_id = 62

        self._0 = ord("0")
        self._9 = ord("9")
        self.a = ord("a")
        self.z = ord("z")
        self.A = ord("A")
        self.Z = ord("Z")

        self.special_tokens = {}

    def __call__(
        self,
        x: str | list[str],
        return_tensors: str | None = None,
        padding: bool = False,
        add_special_tokens: bool = True,
    ) -> torch.Tensor | list[int]:
        assert return_tensors in ["pt", None]

        is_list = isinstance(x, list)
        if not is_list:
            x = [x]

        assert all([isinstance(i, str) for i in x])

        batch_size = len(x)
        sequence_lengths = [len(i) for i in x]
        max_sequence_length = max(sequence_lengths)

        if not padding and batch_size > 1:
            assert all(
                [i == max_sequence_length for i in sequence_lengths]
            ), "padding should be True for examples of unequal shapes"

        y = []
        for sample in x:
            y.append([])
            for token in sample:
                y[-1].append(self._get_token_id(token))

            if add_special_tokens:
                y[-1].append(self.eos_token_id)

        if padding:
            y, attention_mask = pad(inputs=y, pad_token_id=self.pad_token_id)

        if return_tensors == "pt":
            y = torch.tensor(y)
        elif not is_list:
            y = y[0]

        return y

    def _get_token_id(self, x: str) -> None:
        assert isinstance(x, str)

        if len(x) == 1:
            xid = ord(x)

            if self._0 <= xid <= self._9:
                y = xid - self._0
            elif self.a <= xid <= self.z:
                y = xid - self.a + 10
            elif self.A <= xid <= self.Z:
                y = xid - self.A + 36
            else:
                raise ValueError(f"unexpected token ({x})")
        elif x == self.eos_token:
            y = self.eos_token_id
        elif x in self.special_tokens:
            y = self.special_tokens[x]
        else:
            raise ValueError(f"unexpected token ({x})")

        return y

    def add_special_tokens(self, special_tokens: dict) -> None:
        for i, token in enumerate(special_tokens["additional_special_tokens"]):
            self.special_tokens[token] = self.eos_token_id + i + 1
