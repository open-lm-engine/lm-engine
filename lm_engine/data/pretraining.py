# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import logging

from ..accelerator import Accelerator
from ..arguments import TrainingArgs
from ..defaults import INPUT_FORMAT, OUTPUT_FORMAT
from ..logging_utils import log_rank_0
from ..parallel import ProcessGroupManager
from ..tokenizers import TOKENIZER_TYPE
from .dataloader import ResumableDataLoader
from .megatron import GPTDataset, GPTDatasetConfig, MegatronBatchSampler, Split, build, compile_helpers
from .stitched import OrderingStrategy, StitchedDatasetConfig, StitchedSequenceDataset
from .stitched import build_sample_index as build_stitched_sample_index


def _get_train_val_test_samples(
    num_training_steps: int, global_batch_size: int, eval_interval: int, eval_steps: int
) -> tuple[int, int, int]:
    train_samples = num_training_steps * global_batch_size
    val_samples = (num_training_steps // eval_interval + 1) * eval_steps * global_batch_size
    test_samples = eval_steps * global_batch_size

    return train_samples, val_samples, test_samples


def _get_dataloader(
    dataset: GPTDataset | StitchedSequenceDataset | None,
    consumed_samples: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    num_pipeline_stages: int,
    num_workers: int,
):
    if dataset is None:
        return None

    accelerator = Accelerator.get_accelerator()

    if accelerator in [Accelerator.mps, Accelerator.trainium]:
        num_workers = 0

    dataloader = ResumableDataLoader(
        dataset,
        batch_sampler=MegatronBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=(
                micro_batch_size if num_pipeline_stages == 1 else micro_batch_size * gradient_accumulation_steps
            ),
            num_replicas=ProcessGroupManager.get_data_loading_world_size(),
            rank=ProcessGroupManager.get_data_loading_rank(),
        ),
        multiprocessing_context="fork" if accelerator == Accelerator.tpu else None,
        num_workers=num_workers,
        pin_memory=accelerator not in [Accelerator.mps, Accelerator.trainium],
    )

    return iter(dataloader)


def get_pretraining_dataloaders(
    args: TrainingArgs, tokenizer: TOKENIZER_TYPE, consumed_samples: int
) -> tuple[ResumableDataLoader, list[ResumableDataLoader], list[ResumableDataLoader]]:
    assert len(args.datasets) == 1
    class_args = args.datasets[0].class_args

    assert args.datasets[0].max_input_tokens is None
    assert args.datasets[0].max_output_tokens is None
    assert args.datasets[0].input_format == INPUT_FORMAT
    assert args.datasets[0].output_format == OUTPUT_FORMAT

    micro_batch_size = args.training_parameters.micro_batch_size
    gradient_accumulation_steps = args.training_parameters.gradient_accumulation_steps
    num_pipeline_stages = args.distributed_args.num_pipeline_stages
    num_workers = class_args.get("num_workers", 2)

    is_megatron = args.datasets[0].class_name == "MegatronDataset"

    train_samples, val_samples, test_samples = _get_train_val_test_samples(
        args.training_parameters.num_training_steps,
        args.training_parameters.global_batch_size,
        args.training_parameters.eval_interval,
        class_args.get("eval_steps"),
    )

    if is_megatron:
        compile_helpers()

        log_rank_0(logging.INFO, "> building train, validation, and test datasets for GPT ...")

        # Option 1: data loading using --data-path with single file
        # Option 2: data loading using --data-path with multiple weighted files
        # Option 3: data loading using --(train|val|test)-data-path with multiple weighted files
        train_ds, val_ds, test_ds = build(
            sizes=(train_samples, val_samples, test_samples),
            config=GPTDatasetConfig(
                sequence_length=class_args.get("sequence_length"),
                blend=class_args.get("data_path"),
                blend_per_split=[
                    class_args.get("train_data_path"),
                    class_args.get("val_data_path"),
                    class_args.get("test_data_path"),
                ],
                split=class_args.get("split"),
                path_to_cache=class_args.get("data_cache_path"),
                fim_rate=class_args.get("fim_rate", 0),
                fim_spm_rate=class_args.get("fim_spm_rate", 0.5),
            ),
            tokenizer=tokenizer,
            node_uses_local_storage=class_args.get("node_uses_local_storage", False),
            random_seed=class_args.get("seed", args.random_args.seed),
        )
    else:
        assert args.datasets[0].class_name == "StitchedDataset"

        config = StitchedDatasetConfig(
            stitched_seq_path=class_args["stitched_seq_path"],
            tokenized_data_root=class_args["tokenized_data_root"],
            sequence_length=class_args["sequence_length"],
            ordering_strategy=OrderingStrategy(class_args.get("ordering_strategy", "as_stored")),
            seed=class_args.get("seed", args.random_args.seed),
            split_ratio=tuple(class_args.get("split_ratio", [1.0, 0.0, 0.0])),
        )

        caching_allowed = class_args.get("caching_allowed", True)

        def _build_dataset(split: Split, num_samples: int) -> StitchedSequenceDataset | None:
            if num_samples == 0 or config.split_ratio[split.value] == 0.0:
                return None
            stitched_seq, sample_index = build_stitched_sample_index(config, split, num_samples, caching_allowed)
            return StitchedSequenceDataset(config, split, stitched_seq, sample_index)

        train_ds = _build_dataset(Split.train, train_samples)
        val_ds = _build_dataset(Split.valid, val_samples)
        test_ds = _build_dataset(Split.test, test_samples)

        log_rank_0(logging.INFO, "> finished creating StitchedSequenceDataset splits ...")

    log_rank_0(logging.INFO, "> finished creating GPT datasets ...")

    if not isinstance(val_ds, list):
        val_ds = [val_ds]
    if not isinstance(test_ds, list):
        test_ds = [test_ds]

    log_rank_0(logging.INFO, "> finished creating GPT datasets ...")

    kwargs = dict(
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_pipeline_stages=num_pipeline_stages,
        num_workers=num_workers,
    )

    train_ds = _get_dataloader(train_ds, consumed_samples=consumed_samples, **kwargs)
    val_ds = [_get_dataloader(i, consumed_samples=0, **kwargs) for i in val_ds]
    test_ds = [_get_dataloader(i, consumed_samples=0, **kwargs) for i in test_ds]

    return train_ds, val_ds, test_ds
