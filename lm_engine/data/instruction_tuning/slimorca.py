# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from datasets import load_dataset

from ...enums import DatasetSplit
from .base import BaseInstructionDataset


class SlimOrcaDataset(BaseInstructionDataset):
    def prepare_examples(self) -> list[dict]:
        if self.split != DatasetSplit.train:
            return []

        data = load_dataset("Open-Orca/SlimOrca-Dedup")["train"]

        examples = []
        for raw_example in data:
            raw_example = raw_example["conversations"]

            input = self.construct_input_from_format(raw_example[0]["value"], raw_example[1]["value"])
            output = self.construct_output_from_format(raw_example[2]["value"].strip())

            example = self.get_input_output_token_ids(input, output)
            examples.append(example)

        return examples
