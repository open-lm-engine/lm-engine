# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from datasets import load_dataset

from ...enums import DatasetSplit
from .base import BaseInstructionDataset


class AlpacaDataset(BaseInstructionDataset):
    def prepare_examples(self) -> list[dict]:
        if self.split != DatasetSplit.train:
            return []

        data = load_dataset("tatsu-lab/alpaca")["train"]

        examples = []
        for raw_example in data:
            input = self.construct_input_from_format(raw_example["instruction"], raw_example.get("input", ""))
            output = self.construct_output_from_format(raw_example["output"].strip())

            example = self.get_input_output_token_ids(input, output)
            examples.append(example)

        return examples
