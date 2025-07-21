# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from datasets import load_dataset

from ...enums import DatasetSplit
from .base import BaseInstructionDataset


class DollyDataset(BaseInstructionDataset):
    def prepare_examples(self) -> list[dict]:
        if self.split != DatasetSplit.train:
            return []

        data = load_dataset("databricks/databricks-dolly-15k")["train"]

        examples = []
        for raw_example in data:
            input = self.construct_input_from_format(raw_example["instruction"], raw_example.get("context", ""))
            output = self.construct_output_from_format(raw_example["response"].strip())

            example = self.get_input_output_token_ids(input, output)
            examples.append(example)

        return examples
