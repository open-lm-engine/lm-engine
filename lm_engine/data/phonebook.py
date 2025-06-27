# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import itertools
import random
import string

import torch


# $ BOS
# | SEP
# . EOS
# * MASK
class AlphaNumericTokenizer:
    def __init__(self, TO_TOKEN, TO_CHAR):

        self.TO_TOKEN = TO_TOKEN
        self.TO_CHAR = TO_CHAR

        self.bos_token_id = TO_TOKEN["$"]
        self.eos_token_id = TO_TOKEN["."]
        self.sep_token_id = TO_TOKEN["|"]

    def __call__(self, x):
        encoded = [self.TO_TOKEN[c] for c in x]
        return torch.tensor(encoded, dtype=torch.int64)

    def __len__(self):
        return len(self.TO_TOKEN)


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
