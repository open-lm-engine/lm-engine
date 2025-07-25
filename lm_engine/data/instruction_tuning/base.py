# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from ...enums import DatasetSplit
from ...tokenizers import TOKENIZER_TYPE
from ..base import BaseDataset


class BaseInstructionDataset(BaseDataset):
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
    ) -> BaseInstructionDataset:
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
            raise ValueError(f"input_format for {self.__class__.__name__} should be '__input__'")

        self.examples = self.prepare_examples()

    def construct_input_from_format(self, instruction: str, input: str) -> str:
        input_text = instruction + "\n\n"
        if not (input is None or input == ""):
            input_text += f"input: {input}\n"
        input_text += "output:"
        return input_text

    def prepare_examples(self) -> list[dict]:
        raise NotImplementedError()
