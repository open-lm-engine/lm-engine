# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

from transformers import AutoTokenizer

from lm_engine.data import (
    BlendedDatasets,
    BlendedDistributedSampler,
    ResumableDataLoader,
    collate_fn,
    get_datasets_list,
)
from lm_engine.enums import DatasetSplit, Mode

from .test_commons import TestCommons


class DataLoaderTest(TestCommons):
    def test_dataloader_has_correct_order(self) -> None:
        args = TestCommons.load_training_args_for_unit_tests("data_config.yml")
        split = DatasetSplit.train
        mode = Mode.training

        args.datasets[0].class_args["static_examples"] = False
        num_examples = 1000
        args.datasets[0].class_args["num_examples"] = num_examples

        tokenizer = AutoTokenizer.from_pretrained(args.model_args.model_name)
        datasets_list, _ = get_datasets_list(
            dataset_args_list=args.datasets,
            split=split,
            mode=mode,
            tokenizer=tokenizer,
            is_encoder_decoder=False,
        )
        blended_dataset = BlendedDatasets(datasets=datasets_list, split=split)

        world_size = 8
        all_tokens = {}
        for rank in range(world_size):
            sampler = BlendedDistributedSampler(
                dataset=blended_dataset,
                data_sampling_ratios=[1],
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )

            dataloader = ResumableDataLoader(
                blended_dataset,
                batch_size=args.training_parameters.micro_batch_size,
                sampler=sampler,
                collate_fn=partial(
                    collate_fn,
                    mode=mode,
                    loss_mask=args.training_parameters.loss_mask,
                    eos_token_id=tokenizer.eos_token_id,
                    is_encoder_decoder=False,
                    use_padding_free_transformer=args.model_args.use_padding_free_transformer,
                    device="cpu",
                ),
            )

            all_tokens[rank] = []
            for batch in dataloader:
                for example in batch["input_ids"]:
                    all_tokens[rank].append(example[0].item())

            assert len(all_tokens[rank]) == len(set(all_tokens[rank])), f"all tokens were not unique for rank {rank}"

        assert (
            sum([len(all_tokens[rank]) for rank in all_tokens]) == num_examples
        ), "all tokens were not iterated through"
