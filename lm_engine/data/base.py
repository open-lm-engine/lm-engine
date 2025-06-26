# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from transformers import AutoTokenizer

from ..defaults import INPUT_FORMAT, OUTPUT_FORMAT
from ..enums import DatasetSplit, Mode


class BaseDataset(torch.utils.data.Dataset):
    """BaseDataset class to be implemented by all the datasets"""

    def __init__(
        self,
        class_args: dict,
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
        data_name: str,
        input_format: str,
        output_format: str,
        max_input_tokens: int,
        max_output_tokens: int,
    ) -> None:
        super().__init__()

        self.split = split
        self.mode = mode

        self.class_args = class_args

        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder

        self.data_name = data_name
        self.input_format = input_format
        self.output_format = output_format

        # if format is __input__ or __output__ formatting is a no-op
        self.do_format_input = self.input_format != INPUT_FORMAT
        self.do_format_output = self.output_format != OUTPUT_FORMAT

        # length to use for trimming (excludes eos)
        if max_input_tokens is None:
            self.max_input_tokens = None
        else:
            self.max_input_tokens = max_input_tokens

            if self.is_encoder_decoder:
                self.max_input_tokens -= 1

        self.max_output_tokens = None if max_output_tokens is None else max_output_tokens - 1

        self.examples = []

    def construct_input_from_format(self, input: str) -> str:
        """construct input with the specified input_format

        Args:
            input (str): input text

        Returns:
            str: formatted text
        """

        if self.do_format_input:
            return self.input_format.replace(INPUT_FORMAT, input, 1)
        return input

    def construct_output_from_format(self, output: str) -> str:
        """construct output with the specified output_format

        Args:
            output (str): output text

        Returns:
            str: formatted text
        """

        if self.do_format_output:
            return self.output_format.replace("__output__", output, 1)
        return output

    def get_input_output_token_ids(self, input: str, output: str) -> dict:
        """tokenizes the input and output text

        Args:
            input (str): input text
            output (str): output text

        Returns:
            dict: an example
        """

        eos_token_id: int = self.tokenizer.eos_token_id

        input: list[int] = self.tokenizer(input, add_special_tokens=False)["input_ids"]

        if self.is_encoder_decoder:
            if self.max_input_tokens is not None:
                input = input[: self.max_input_tokens - 1]
            input.append(eos_token_id)
        else:
            if self.max_input_tokens is not None:
                input = input[: self.max_input_tokens]

        if self.mode == Mode.training:
            output: list[int] = self.tokenizer(output, add_special_tokens=False)["input_ids"]

            if self.max_output_tokens is not None:
                output = output[: self.max_output_tokens - 1]
            output.append(eos_token_id)

            if not self.is_encoder_decoder:
                input.extend(output)

            result = {"input": input, "output": output}
        else:
            result = {"input": input}

        return result

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        return

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)


class BlendedDatasets(torch.utils.data.Dataset):
    """Concatenated list of datasets for training or inference"""

    def __init__(self, datasets: list[BaseDataset], split: DatasetSplit) -> None:
        super().__init__()

        self.split = split
        self.datasets = datasets

        self.num_examples = sum(self.get_num_examples_in_each_dataset())
        self.indexing_array = self._get_indexing_array()

    def get_num_datasets(self) -> int:
        """returns the number of datasets in the mixture

        Returns:
            int: number of datasets in the mixture
        """

        return len(self.datasets)

    def get_num_examples_in_each_dataset(self) -> list[int]:
        """returns the number of examples in each dataset component

        Returns:
            list[int]: the number of examples in each dataset component
        """

        return [len(dataset) for dataset in self.datasets]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        return

    def _get_indexing_array(self) -> list[tuple[int]]:
        num_examples_in_each_dataset = self.get_num_examples_in_each_dataset()

        indexing_array = []
        for dataset_index, num_examples in enumerate(num_examples_in_each_dataset):
            for example_id in range(num_examples):
                indexing_array.append((dataset_index, example_id))

        return indexing_array

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> dict:
        dataset_index, example_index = self.indexing_array[index]
        example = self.datasets[dataset_index][example_index]
        return example

    def __repr__(self) -> str:
        x = f"number of datasets = {self.get_num_datasets()}\n"
        x += f"total examples in the entire dataset mixture = {len(self)}"

        for dataset in self.datasets:
            x += f"\nexamples in {dataset.__class__.__name__} ({dataset.data_name}) = {len(dataset)}"

        return x
