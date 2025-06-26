# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import json
import os
import tempfile

from parameterized import parameterized
from transformers import AutoTokenizer

from lm_engine.data import get_datasets_list
from lm_engine.enums import DatasetSplit, Mode

from .test_commons import TestCommons


class JSONLinesTest(TestCommons):
    @parameterized.expand([DatasetSplit.train, DatasetSplit.val, DatasetSplit.test])
    def test_jsonlines_loads(self, split: DatasetSplit) -> None:
        args = TestCommons.load_training_args_for_unit_tests("data_config.yml")
        mode = Mode.training

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = os.path.join(tmpdir, split.value)
            os.makedirs(split_dir)

            for i in range(3):
                filename = os.path.join(split_dir, f"filename{i}.jsonl")
                open(filename, "w").writelines([json.dumps({"input": str(j), "output": str(j)}) for j in range(5)])

            args.datasets[0].class_name = "HuggingFaceDataset"
            args.datasets[0].class_args["data_path"] = tmpdir

            tokenizer = AutoTokenizer.from_pretrained(args.model_args.model_name)
            datasets_list, _ = get_datasets_list(
                dataset_args_list=args.datasets,
                split=split,
                mode=mode,
                tokenizer=tokenizer,
                is_encoder_decoder=False,
            )

        assert len(datasets_list) == 1
        assert len(datasets_list[0]) == 15
