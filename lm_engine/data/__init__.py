# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging
from functools import partial

from ..arguments import DatasetArgs, TrainingArgs
from ..enums import DatasetSplit, Mode
from ..tokenizers import TOKENIZER_TYPE
from ..utils import ProcessGroupManager, log_rank_0, run_rank_n
from .base import BaseDataset, BlendedDatasets
from .dataloader import ResumableDataLoader
from .debug import DebugDataset
from .huggingface import HuggingFaceDataset
from .ibm import get_ibm_dataloaders
from .instruction_tuning import AlpacaDataset, DollyDataset, SlimOrcaDataset
from .megatron import get_megatron_gpt_dataloaders
from .sampler import BlendedDistributedSampler
from .sst2 import SST2Dataset
from .utils import collate_fn, custom_iterator, get_next_batch


_DATASETS_LIST = {
    "AlpacaDataset": AlpacaDataset,
    "DebugDataset": DebugDataset,
    "DollyDataset": DollyDataset,
    "HuggingFaceDataset": HuggingFaceDataset,
    "SlimOrcaDataset": SlimOrcaDataset,
    "SST2Dataset": SST2Dataset,
}


def get_datasets_list(
    dataset_args_list: list[DatasetArgs], split: DatasetSplit, mode: Mode, tokenizer: TOKENIZER_TYPE
) -> tuple[list[BaseDataset], list[int]]:
    """get the list of datasets from their configs

    Args:
        dataset_args_list (list[DatasetArgs]): list of DatasetArgs objects
        split (DatasetSplit): train / val / test split
        mode (Mode): training / inference mode for running the program
        tokenizer (TOKENIZER_TYPE): tokenizer

    Raises:
        ValueError: if invalid class_name for dataset is found

    Returns:
        tuple[List[BaseDataset], list[int]]: tuple of list of datasets and the respective dataset sampling ratios
    """

    datasets_list = []
    data_sampling_ratios = []
    for data_args in dataset_args_list:
        if data_args.class_name not in _DATASETS_LIST:
            raise ValueError(f"invalid class_name ({data_args.class_name}) for dataset")

        dataset = _DATASETS_LIST[data_args.class_name](
            class_args=data_args.class_args,
            split=split,
            mode=mode,
            tokenizer=tokenizer,
            data_name=data_args.data_name,
            input_format=data_args.input_format,
            output_format=data_args.output_format,
            max_input_tokens=data_args.max_input_tokens,
            max_output_tokens=data_args.max_output_tokens,
        )

        if len(dataset) > 0:
            datasets_list.append(dataset)
            data_sampling_ratios.append(data_args.data_sampling_ratio)

            log_rank_0(
                logging.INFO, f"examples in {dataset.__class__.__name__} ({data_args.data_name}) = {len(dataset)}"
            )

    assert all([i is not None for i in data_sampling_ratios]) or all(
        [i is None for i in data_sampling_ratios]
    ), "either all data_sampling_ratios should be specified or all should be None"
    if all([i is None for i in data_sampling_ratios]):
        data_sampling_ratios = [len(i) for i in datasets_list]

    return datasets_list, data_sampling_ratios


def get_finetuning_dataloader(
    args: TrainingArgs, split: DatasetSplit, mode: Mode, tokenizer: TOKENIZER_TYPE
) -> ResumableDataLoader:
    """prepares datasets and sampler

    Args:
        args (TrainingArgs): arguments based on training / inference mode
        split (DatasetSplit): train / val / test split
        mode (Mode): training / inference mode
        tokenizer (TOKENIZER_TYPE): tokenizer

    Returns:
        ResumableDataLoader: dataloader for a blended dataset
    """

    assert mode == Mode.training, "blended dataset is only supported in training mode"

    if ProcessGroupManager.get_tensor_parallel_rank() != 0:
        return

    micro_batch_size = args.training_parameters.micro_batch_size

    datasets_list, data_sampling_ratios = get_datasets_list(
        dataset_args_list=args.datasets, split=split, mode=Mode.training, tokenizer=tokenizer
    )

    if len(datasets_list) == 0:
        return None

    blended_dataset = BlendedDatasets(datasets=datasets_list, split=split)

    # routing to data parallel worker is done by sampler
    sampler = BlendedDistributedSampler(
        dataset=blended_dataset,
        data_sampling_ratios=[1] if len(datasets_list) == 1 else data_sampling_ratios,
        num_replicas=ProcessGroupManager.get_data_parallel_world_size(),
        rank=ProcessGroupManager.get_data_parallel_rank(),
        ignore_sampling_proportion_for_validation=args.training_parameters.ignore_sampling_proportion_for_validation,
        shuffle=split == DatasetSplit.train,
        seed=args.random_args.seed,
        drop_last=False,
    )

    # dataloader is unaware of data parallel routing
    dataloader = ResumableDataLoader(
        blended_dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        collate_fn=partial(
            collate_fn,
            mode=mode,
            loss_mask=args.training_parameters.loss_mask,
            eos_token_id=tokenizer.eos_token_id,
            use_padding_free_transformer=args.model_args.use_padding_free_transformer,
            pad_to_multiple_of=ProcessGroupManager.get_tensor_parallel_world_size(),
        ),
    )

    _log_dataset(
        blended_dataset=blended_dataset,
        sampler=sampler,
        split=split,
        num_training_steps=args.training_parameters.num_training_steps,
        gradient_accumulation_steps=args.training_parameters.gradient_accumulation_steps,
        micro_batch_size=args.training_parameters.micro_batch_size,
    )

    return dataloader


def get_pretraining_dataloaders(
    args: TrainingArgs, tokenizer: TOKENIZER_TYPE, consumed_samples: int
) -> tuple[ResumableDataLoader, list[ResumableDataLoader], list[ResumableDataLoader]]:
    if args.datasets[0].class_name == "MegatronDataset":
        dataloaders = get_megatron_gpt_dataloaders(args, tokenizer, consumed_samples=consumed_samples)
    elif args.datasets[0].class_name == "IBMDataset":
        dataloaders = get_ibm_dataloaders(args, tokenizer)

    return dataloaders


@run_rank_n
def _log_dataset(
    blended_dataset: BlendedDatasets,
    sampler: BlendedDistributedSampler,
    split: DatasetSplit,
    num_training_steps: int,
    gradient_accumulation_steps: int,
    micro_batch_size: int,
) -> None:
    log_rank_0(logging.INFO, f"{'-' * 25} {split.value} {'-' * 25}")
    log_rank_0(logging.INFO, blended_dataset)

    dp_world_size = ProcessGroupManager.get_data_parallel_world_size()

    if split == DatasetSplit.train:
        total_samples_seen = num_training_steps * gradient_accumulation_steps * micro_batch_size * dp_world_size
    else:
        num_steps = len(blended_dataset) // (micro_batch_size * dp_world_size)
        if len(blended_dataset) % (micro_batch_size * dp_world_size) != 0:
            num_steps += 1

        total_samples_seen = num_steps * micro_batch_size * dp_world_size

    log_rank_0(logging.INFO, "*" * 57)
    log_rank_0(logging.INFO, f"total samples seen = {total_samples_seen}")
    log_rank_0(logging.INFO, f"total epochs for the dataset mixture = {total_samples_seen / len(blended_dataset)}")
    log_rank_0(logging.INFO, sampler)
    log_rank_0(logging.INFO, "-" * 57)
