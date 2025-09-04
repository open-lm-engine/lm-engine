# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import itertools
import random
import string

from tqdm import trange

from ..enums import DatasetSplit, Mode
from ..tokenizers import TOKENIZER_TYPE
from .base import BaseDataset


class PhonebookDataset(BaseDataset):
    def __init__(
        self,
        class_args: dict,
        split: DatasetSplit,
        mode: Mode,
        tokenizer: TOKENIZER_TYPE,
        data_name: str,
        input_format: str,
        output_format: str,
        max_input_tokens: int,
        max_output_tokens: int,
    ) -> PhonebookDataset:
        super().__init__(
            class_args=class_args,
            split=split,
            mode=mode,
            tokenizer=tokenizer,
            data_name=data_name,
            input_format=input_format,
            output_format=output_format,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )

        self.separator_token = "<sep>"

        assert not self.do_format_input
        assert not self.do_format_output
        assert self.max_input_tokens is None
        assert self.max_output_tokens is None
        assert self.separator_token in tokenizer.get_vocab()

        name_length = self.class_args["name_length"]
        num_digits = self.class_args["num_digits"]
        seed = self.class_args.get("seed", 42)

        num_total_names = 26**name_length
        num_phone_numbers = 10**num_digits

        self.phonebook_size = self.class_args.get("phonebook_size", min(num_total_names, num_phone_numbers))

        assert (
            min(num_total_names, num_phone_numbers) >= self.phonebook_size
        ), f"either {num_total_names} or {num_phone_numbers} is too small!"

        names = list(itertools.product(list(string.ascii_lowercase), repeat=name_length))
        phone_numbers = list(itertools.product(list(string.digits), repeat=num_digits))

        local_random = random.Random(seed)
        local_random.shuffle(names)
        local_random.shuffle(phone_numbers)

        names = names[: self.phonebook_size]
        phone_numbers = phone_numbers[: self.phonebook_size]

        self.examples = []
        for i in trange(self.phonebook_size):
            sample = "".join(names[i]) + self.separator_token + "".join(phone_numbers[i])
            sample = tokenizer(sample, add_special_tokens=False)
            sample += [tokenizer.eos_token_id]

            self.examples.append(sample)

    def __len__(self) -> int:
        return self.phonebook_size
