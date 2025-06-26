# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from copy import deepcopy

from transformers import AutoTokenizer

from lm_engine.data import BlendedDatasets, get_datasets_list
from lm_engine.enums import DatasetSplit, Mode

from .test_commons import TestCommons


class BlendedDatasetsTest(TestCommons):
    def test_dataloader(self) -> None:
        args = TestCommons.load_training_args_for_unit_tests("data_config.yml")
        split = DatasetSplit.train
        mode = Mode.training

        for i in range(1, 4):
            dataset = deepcopy(args.datasets[0])

            dataset.data_name = f"debug{i}"
            dataset.class_args["num_examples"] = (i + 1) * 100
            dataset.class_args["token_id"] = i

            args.datasets.append(dataset)

        tokenizer = AutoTokenizer.from_pretrained(args.model_args.model_name)

        datasets_list, _ = get_datasets_list(
            dataset_args_list=args.datasets,
            split=split,
            mode=mode,
            tokenizer=tokenizer,
            is_encoder_decoder=False,
        )

        blended_dataset = BlendedDatasets(datasets=datasets_list, split=split)

        num_examples_in_each_dataset = [100, 200, 300, 400]
        assert blended_dataset.get_num_examples_in_each_dataset() == num_examples_in_each_dataset

        blended_dataset = iter(blended_dataset)

        for dataset_index, num_examples in enumerate(num_examples_in_each_dataset):
            for _ in range(num_examples):
                example = next(blended_dataset)

                assert len(example["input"]) == 1024
                assert len(example["output"]) == 128
                assert all([i == dataset_index for i in example["input"]])
                assert all([i == dataset_index for i in example["output"]])
