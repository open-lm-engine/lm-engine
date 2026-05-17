# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import os

import torch

from ...utils import compile_cpp_extension


class AlphaNumericTokenizer:
    def __init__(self) -> AlphaNumericTokenizer:
        current_dir = os.path.dirname(__file__)
        self._tokenizer = compile_cpp_extension(
            "alpha_numeric_cpp",
            sources=os.path.join(current_dir, "alpha_numeric.cpp"),
            build_directory=os.path.join(current_dir, "build"),
            extra_cflags=["-O3", "-Wall", "-shared", "-std=c++14", "-fPIC", "-fdiagnostics-color"],
        ).AlphaNumericTokenizer()

    def __call__(
        self,
        x: str | list[str],
        return_tensors: str | None = None,
        padding: bool = False,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor | list[int] | list[list[int]]]:
        assert return_tensors in ["pt", None]

        is_list = isinstance(x, list)
        if not is_list:
            x = [x]

        assert all([isinstance(i, str) for i in x])

        if not padding and len(x) > 1:
            lengths = [len(s) for s in x]
            assert all([l == lengths[0] for l in lengths]), "padding should be True for examples of unequal shapes"

        result = self._tokenizer.encode_batch(x, padding, add_special_tokens)
        result = {"input_ids": result.input_ids, "attention_mask": result.attention_mask}

        if return_tensors == "pt":
            for k, v in result.items():
                result[k] = torch.tensor(v)

        if not is_list:
            for k, v in result.items():
                result[k] = result[k][0]

        return result

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens)

    def batch_decode(self, ids: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.batch_decode(ids, skip_special_tokens)
