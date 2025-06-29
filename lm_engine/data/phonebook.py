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


class PhonebookDataset(BaseDataset):
    def __init__(
        self, tokenizer: TOKENIZER_TYPE, phonebook_size: int, name_length: int, num_digits: int
    ) -> PhonebookDataset:
        num_total_names = 26**name_length
        num_phone_numbers = 10**num_digits

        assert (
            min(num_total_names, num_phone_numbers) >= phonebook_size
        ), f"either {num_total_names} or {num_phone_numbers} is too small!"

        names = list(itertools.product(list(string.ascii_lowercase), repeat=name_length))
        phone_numbers = list(itertools.product(list(string.digits), repeat=num_digits))

        random.shuffle(names)
        random.shuffle(phone_numbers)

        names = names[:phonebook_size]
        phone_numbers = phone_numbers[:phonebook_size]

        self.dataset = []
        for i in range(phonebook_size):
            sample = "".join(names[i]) + "<sep>" + "".join(phone_numbers[i])
            sample = tokenizer(sample, add_special_tokens=False)
            sample += [tokenizer.eos_token_id]

            self.dataset.append(sample)

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
