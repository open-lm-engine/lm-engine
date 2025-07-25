# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from datasets import load_dataset

from ..enums import DatasetSplit
from ..tokenizers import TOKENIZER_TYPE
from .base import BaseDataset


class SST2Dataset(BaseDataset):
    """SST2 dataset for sentiment classification"""

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
    ) -> SST2Dataset:
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

        self.examples = self.prepare_examples()

    def prepare_examples(self) -> list[dict]:
        split = self.split.value
        if split == "val":
            split = "validation"

        raw_examples = load_dataset("sst2")[split]
        examples = []
        for raw_example in raw_examples:
            input = self.construct_input_from_format(raw_example["sentence"].strip())
            output = (
                self.construct_output_from_format("positive" if raw_example["label"] == 1 else "negative")
                if self.use_output
                else None
            )

            example = self.get_input_output_token_ids(input, output)
            examples.append(example)

        return examples
