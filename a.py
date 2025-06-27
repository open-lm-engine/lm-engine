# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import itertools
import random
import string

import torch


class AlphaNumericTokenizer:
    def __init__(self, lowercase_only: bool = True) -> AlphaNumericTokenizer:
        self.eos_token = "<|endoftext|>"
        self.pad_token = self.eos_token
        self.lowercase_only = lowercase_only

        if not self.lowercase_only:
            self.A = ord("A")
            self.Z = ord("Z")

        self.a = ord("a")
        self.z = ord("z")
        self._0 = ord("0")
        self._9 = ord("9")

        self.special_tokens = {}

    def __call__(
        self, x: str | list[str], return_tensors: str | None = None, padding: bool = False
    ) -> torch.Tensor | list[int]:
        is_list = isinstance(x, list)

        if is_list:
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
                y[-1].append(self.get_token_id(token))

        if return_tensors == "pt":
            y = torch.tensor(y)
        elif not is_list:
            y = y[0]

        return y

    def get_token_id(self, x: str) -> None:
        assert len(x) == 1

        is_number = self._0 <= x <= self._9

        if is_number:
            y = x - self._0
        else:
            is_lowercase_alphabet = self.a <= x <= self.z

            if is_lowercase_alphabet:
                y = x - self.a + 10
            else:
                is_special_token = x in [self.special_tokens]

                if is_special_token:
                    pass
                else:
                    assert not self.lowercase_only
                    is_uppercase_alphabet = self.A <= x <= self.Z

                    if is_uppercase_alphabet:
                        y = x - self.A + 36
                    else:
                        raise ValueError(f"invalid token ({x})")

        return y


def sample_phonebook(tokenizer, phonebook_size=500_000, name_length=5, phone_digits=8):
    assert (
        min(26**name_length, 10**phone_digits) >= phonebook_size
    ), f"either {26 ** name_length} or {10 ** phone_digits} is too small!"

    name_iter = list(itertools.product(list(string.ascii_lowercase), repeat=name_length))
    print(name_iter)
    phone_iter = list(itertools.product(list(string.digits), repeat=phone_digits))

    print("Shuffle name!\n")
    random.shuffle(name_iter)

    print("Shuffle phone!\n")
    random.shuffle(phone_iter)

    ret = []
    for i, name in enumerate(name_iter):
        if i == phonebook_size:
            # stack and shuffle
            return torch.vstack(ret)
        ret.append(tokenizer(f'${"".join(name)}|{"".join(phone_iter[i])}.'))
    return torch.vstack(ret)


tokenizer = AlphaNumericTokenizer()
# print(sample_phonebook(tokenizer))

x = "abcd: 123"
print(x)
print(tokenizer(x))
