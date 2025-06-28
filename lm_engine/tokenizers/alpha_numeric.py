# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import itertools
import random
import string

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
        assert len(x) == 1

        xid = ord(x)

        if self._0 <= xid <= self._9:
            y = xid - self._0
        elif self.a <= xid <= self.z:
            y = xid - self.a + 10
        elif self.A <= xid <= self.Z:
            y = xid - self.A + 36
        elif xid == self.eos_token:
            y = self.eos_token_id
        else:
            raise ValueError(f"unexpected token ({x})")

        return y


def sample_phonebook(args, tokenizer, phonebook_size=500_000, name_length=5, phone_digits=8):
    assert (
        min(26**name_length, 10**phone_digits) >= phonebook_size
    ), f"either {26 ** name_length} or {10 ** phone_digits} is too small!"

    name_iter = list(itertools.product(list(string.ascii_lowercase), repeat=name_length))
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


class PhonebookDataset(torch.utils.data.Dataset):
    def __init__(self, phonebook_dict):
        self.phonebook_dict = phonebook_dict

    def __len__(self):
        return len(self.phonebook_dict["input_ids"])

    def __getitem__(self, idx):
        if "global_token_unique_index" in self.phonebook_dict:
            return {
                "input_ids": self.phonebook_dict["input_ids"][idx],
                "mask": self.phonebook_dict["mask"][idx],
                "global_seq_unique_index": self.phonebook_dict["global_seq_unique_index"][idx],
                "global_token_unique_index": self.phonebook_dict["global_token_unique_index"][idx],
                "global_token_per_seq_index": self.phonebook_dict["global_token_per_seq_index"][idx],
                "global_seq_random_unique_number": self.phonebook_dict["global_seq_random_unique_number"][idx],
                "global_token_random_unique_number": self.phonebook_dict["global_token_random_unique_number"][idx],
            }
        else:
            return {"input_ids": self.phonebook_dict["input_ids"][idx], "mask": self.phonebook_dict["mask"][idx]}


# $ BOS
# | SEP
# . EOS
def get_tokenizer(args):
    letters = dict(zip(string.ascii_lowercase, range(26)))
    digits = dict(zip(string.digits, range(26, 36)))
    TO_TOKEN = {**letters, **digits}

    symbols = {"$": len(TO_TOKEN), "|": len(TO_TOKEN) + 1, ".": len(TO_TOKEN) + 2}
    TO_TOKEN = {**symbols, **TO_TOKEN}

    TO_CHAR = {v: k for k, v in TO_TOKEN.items()}

    tokenizer = NumberTokenizer(TO_TOKEN, TO_CHAR)
    return tokenizer, TO_TOKEN, TO_CHAR
