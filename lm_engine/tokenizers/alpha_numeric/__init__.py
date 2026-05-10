# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load as load_cpp_extension


_dir = os.path.dirname(__file__)
_build_dir = os.path.join(_dir, "build")
os.makedirs(_build_dir, exist_ok=True)

_MODULE = load_cpp_extension(
    "alpha_numeric_cpp",
    sources=[os.path.join(_dir, "alpha_numeric.cpp")],
    extra_cflags=["-O3", "-Wall", "-shared", "-std=c++14", "-fPIC", "-fdiagnostics-color"],
    build_directory=_build_dir,
    verbose=False,
)


class AlphaNumericTokenizer:
    def __init__(self, lowercase_only: bool = True) -> AlphaNumericTokenizer:
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = 62

        self.pad_token = self.eos_token
        self.pad_token_id = self.eos_token_id

        self.lowercase_only = lowercase_only
        self.special_tokens = {}

        self._tokenizer = _MODULE.AlphaNumericTokenizer(lowercase_only)

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

        if not padding and len(x) > 1:
            lengths = [len(s) for s in x]
            assert all([l == lengths[0] for l in lengths]), "padding should be True for examples of unequal shapes"

        result = self._tokenizer.encode_batch(x, padding, add_special_tokens)
        x = result.input_ids

        if return_tensors == "pt":
            x = torch.tensor(x)
        elif not is_list:
            x = x[0]

        return x

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens)

    def batch_decode(self, ids: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.batch_decode(ids, skip_special_tokens)
