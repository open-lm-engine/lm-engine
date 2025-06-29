# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import itertools
import random
import string

import torch

from ..tokenizers import TOKENIZER_TYPE
from .base import BaseDataset


def sample_phonebook(tokenizer: TOKENIZER_TYPE, phonebook_size: int, name_length: int = 5, phone_digits: int = 8):
    num_total_names = 26**name_length
    num_phone_numbers = 10**phone_digits

    assert (
        min(num_total_names, num_phone_numbers) >= phonebook_size
    ), f"either {num_total_names} or {num_phone_numbers} is too small!"

    name_iter = list(itertools.product(list(string.ascii_lowercase), repeat=name_length))
    phone_iter = list(itertools.product(list(string.digits), repeat=phone_digits))

    random.shuffle(name_iter)
    random.shuffle(phone_iter)

    ret = []
    for i, name in enumerate(name_iter):
        if i == phonebook_size:
            # stack and shuffle
            return torch.vstack(ret)
        ret.append(tokenizer(f'${"".join(name)}|{"".join(phone_iter[i])}.'))
    return torch.vstack(ret)


class PhonebookDataset(BaseDataset):
    def __init__(
        self, tokenizer: TOKENIZER_TYPE, phonebook_size: int, name_length: int = 5, num_digits: int = 8
    ) -> PhonebookDataset:
        num_total_names = 26**name_length
        num_phone_numbers = 10**num_digits

        assert (
            min(num_total_names, num_phone_numbers) >= phonebook_size
        ), f"either {num_total_names} or {num_phone_numbers} is too small!"

        name_iter = list(itertools.product(list(string.ascii_lowercase), repeat=name_length))
        phone_iter = list(itertools.product(list(string.digits), repeat=num_digits))

        random.shuffle(name_iter)
        random.shuffle(phone_iter)

        ret = []
        for i, name in enumerate(name_iter):
            if i == phonebook_size:
                # stack and shuffle
                return torch.vstack(ret)
            ret.append(tokenizer(f'${"".join(name)}|{"".join(phone_iter[i])}.'))
        return torch.vstack(ret)

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
