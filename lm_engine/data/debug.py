# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from ..enums import DatasetSplit
from ..tokenizers import TOKENIZER_TYPE
from .base import BaseDataset


class DebugDataset(BaseDataset):
    """A dummy dataset for profiling and timing the code"""

    def __init__(
        self,
        class_args: dict,
        split: DatasetSplit,
        use_output: bool,
        tokenizer: TOKENIZER_TYPE,
        data_name: str,
        input_format: str,
        output_format: str,
        max_input_tokens: int,
        max_output_tokens: int,
    ) -> DebugDataset:
        super().__init__(
            class_args=class_args,
            split=split,
            use_output=use_output,
            tokenizer=tokenizer,
            data_name=data_name,
            input_format=input_format,
            output_format=output_format,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )

        if self.do_format_input:
            raise ValueError("DebugDataset does not support input formatting")
        if self.do_format_output:
            raise ValueError("DebugDataset does not support output formatting")

        self._length = class_args.get("num_examples")
        assert isinstance(self._length, int) and self._length > 0

        self._token_id = class_args.get("token_id", self.tokenizer.eos_token_id)
        self._static_examples = class_args.get("static_examples", True)

        if self._static_examples:
            self._example = self._get_example(self._token_id)

    def _get_example(self, token_id: int) -> dict:
        example = {"input": [token_id] * self.max_input_tokens}

        if self.use_output:
            example["output"] = [token_id] * (self.max_output_tokens + 1)

        return example

    def __getitem__(self, index: int) -> dict:
        return self._example if self._static_examples else self._get_example(index)

    def __len__(self) -> int:
        return self._length
