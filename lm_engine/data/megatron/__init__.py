import logging

import torch
import torch.distributed
from transformers import AutoTokenizer

from ...arguments import TrainingArgs
from ...defaults import INPUT_FORMAT, OUTPUT_FORMAT
from ...utils import ProcessGroupManager, log_rank_0
from ..dataloader import DispatchingDataLoader, ResumableDataLoader, get_source_and_broadcast_group
from .blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from .blended_megatron_dataset_config import GPTDatasetConfig
from .gpt_dataset import GPTDataset
from .sampler import MegatronBatchSampler
from .utils import Split, compile_helpers


def get_megatron_gpt_dataloaders(args: TrainingArgs, tokenizer: AutoTokenizer, consumed_samples: int) -> None:
    assert len(args.datasets) == 1
    class_args = args.datasets[0].class_args

    assert args.datasets[0].max_input_tokens is None
    assert args.datasets[0].max_output_tokens is None
    assert args.datasets[0].max_output_tokens is None
    assert args.datasets[0].input_format == INPUT_FORMAT
    assert args.datasets[0].output_format == OUTPUT_FORMAT

    micro_batch_size = args.training_parameters.micro_batch_size
    gradient_accumulation_steps = args.training_parameters.gradient_accumulation_steps
    num_pipeline_stages = args.distributed_args.num_pipeline_stages
    sequence_length = class_args.get("sequence_length")

    compile_helpers()

    log_rank_0(logging.INFO, "> building train, validation, and test datasets for GPT ...")

    dispatching_dataloader = args.distributed_args.dispatching_dataloader

    if dispatching_dataloader:
        assert (
            ProcessGroupManager.get_tensor_parallel_world_size() == 1
        ), "tensor parallel doesn't support dispatching dataloader"

        num_ranks_per_node = torch.cuda.device_count()
        node_rank = ProcessGroupManager.get_global_rank() // num_ranks_per_node
        num_nodes = ProcessGroupManager.get_world_size() // num_ranks_per_node

        def _get_source_broadcast_mapping() -> dict:
            result = {}
            for i in range(num_nodes):
                source = i * num_ranks_per_node
                ranks = list(range(source, source + num_ranks_per_node))
                result[source] = torch.distributed.new_group(ranks)
            return result

        source_broadcast_mapping = _get_source_broadcast_mapping()

        # only build dataloader on first rank of each node
        is_built_on_rank = ProcessGroupManager.get_global_rank() == node_rank * num_ranks_per_node
    else:
        # only build dataloader on first rank of each TP group
        is_built_on_rank = ProcessGroupManager.is_tensor_parallel_first_rank()

    gpt_dataset_builder = BlendedMegatronDatasetBuilder(
        GPTDataset,
        sizes=_get_train_val_test_samples(
            args.training_parameters.num_training_steps,
            micro_batch_size,
            gradient_accumulation_steps,
            args.training_parameters.eval_interval,
            class_args.get("eval_steps"),
        ),
        config=GPTDatasetConfig(
            # the dataset is None if is_built_on_rank is False
            is_built_on_rank=is_built_on_rank,
            random_seed=class_args.get("seed", args.random_args.seed),
            sequence_length=sequence_length,
            blend=class_args.get("data_path"),
            blend_per_split=[
                class_args.get("train_data_path"),
                class_args.get("val_data_path"),
                class_args.get("test_data_path"),
            ],
            split=class_args.get("split"),
            path_to_cache=class_args.get("data_cache_path"),
            return_document_ids=False,
            fim_rate=class_args.get("fim_rate", 0),
            fim_spm_rate=class_args.get("fim_spm_rate", 0.5),
            node_uses_local_storage=class_args.get("node_uses_local_storage", False),
        ),
        tokenizer=tokenizer,
    )

    data_path = class_args.get("data_path")
    train_data_path = class_args.get("train_data_path")
    train_weighted_split_paths = class_args.get("train_weighted_split_paths")

    # Option 1: data loading using --data-path with single file
    # Option 2: data loading using --data-path with multiple weighted files
    # Option 3: data loading using --(train|val|test)-data-path with multiple weighted files
    if data_path is not None or train_data_path is not None:
        train_ds, val_ds, test_ds = gpt_dataset_builder.build()

        if not isinstance(val_ds, list):
            val_ds = [val_ds]
        if not isinstance(test_ds, list):
            test_ds = [test_ds]

    # Option 4: data loading using --(train|val|test)-weighted-split-paths
    elif train_weighted_split_paths:

        def _parse_and_get_dataset(weighted_split_paths: list[dict], dataset_split: Split) -> list[GPTDataset]:
            if weighted_split_paths is None:
                return []

            names = []
            paths = []
            splits = []
            weights = []
            for group in weighted_split_paths:
                assert len(group) == 1
                group_name = list(group.keys())[0]
                datasets_splits_list = group[group_name]

                names_ = []
                paths_ = []
                splits_ = []
                weights_ = []
                for d in datasets_splits_list:
                    names_.append(group_name)
                    paths_.append(d["path"])
                    splits_.append(d["split"])
                    weights_.append(d["weight"])

                names.append(names_)
                paths.append(paths_)
                splits.append(splits_)
                weights.append(weights_)

            return gpt_dataset_builder.build_dataset_single_split(names, paths, splits, weights, dataset_split)

        assert len(train_weighted_split_paths) == 1, "only 1 dataset group can be passed for training"
        train_ds = _parse_and_get_dataset(train_weighted_split_paths, Split.train)[0]

        val_ds = _parse_and_get_dataset(class_args.get("val_weighted_split_paths"), Split.valid)
        test_ds = _parse_and_get_dataset(class_args.get("test_weighted_split_paths"), Split.test)
    else:
        raise NotImplementedError("No dataloading argument passed")

    log_rank_0(logging.INFO, "> finished creating GPT datasets ...")

    def _get_dataloader(dataset: GPTDataset | None, consumed_samples: int):
        # we use batch sampler here to match the data order of NVIDIA's megatron repo
        if dispatching_dataloader:
            is_dataset_none_on_source_rank = [dataset is None if is_built_on_rank else False]
            _, _source_rank, _, _broadcast_group = get_source_and_broadcast_group(source_broadcast_mapping)
            torch.distributed.broadcast_object_list(
                is_dataset_none_on_source_rank, src=_source_rank, group=_broadcast_group
            )
            is_dataset_none_on_source_rank = is_dataset_none_on_source_rank[0]

            if is_dataset_none_on_source_rank:
                return None

            if is_built_on_rank:
                assert dataset is not None, "dataset shouldn't be None when is_built_on_rank is True"

                batch_sampler = MegatronBatchSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=(
                        micro_batch_size * num_ranks_per_node
                        if num_pipeline_stages == 1
                        else micro_batch_size * gradient_accumulation_steps * num_ranks_per_node
                    ),
                    num_replicas=num_nodes,
                    rank=node_rank,
                )
            else:
                assert dataset is None, "dataset should be None when is_built_on_rank is False"

                batch_sampler = None

            dataloader = DispatchingDataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=class_args.get("num_workers", 2),
                pin_memory=True,
                source_broadcast_mapping=source_broadcast_mapping,
                broadcast_world_size=num_ranks_per_node,
                static_shape_per_rank=(
                    (micro_batch_size if num_pipeline_stages == 1 else micro_batch_size * gradient_accumulation_steps),
                    sequence_length + 1,
                ),
                keys=["text"],
            )
        else:
            if dataset is None:
                return None

            batch_sampler = MegatronBatchSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=(
                    micro_batch_size if num_pipeline_stages == 1 else micro_batch_size * gradient_accumulation_steps
                ),
                num_replicas=ProcessGroupManager.get_data_parallel_world_size(),
                rank=ProcessGroupManager.get_data_parallel_rank(),
            )

            dataloader = ResumableDataLoader(
                dataset, batch_sampler=batch_sampler, num_workers=class_args.get("num_workers", 2), pin_memory=True
            )

        return iter(dataloader)

    train_ds = _get_dataloader(train_ds, consumed_samples)
    val_ds = [_get_dataloader(i, 0) for i in val_ds]
    test_ds = [_get_dataloader(i, 0) for i in test_ds]

    return train_ds, val_ds, test_ds


def _get_train_val_test_samples(
    num_training_steps: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    eval_interval: int,
    eval_steps: int,
) -> tuple[int]:
    dp_world_size = ProcessGroupManager.get_data_parallel_world_size()

    train_samples = num_training_steps * micro_batch_size * gradient_accumulation_steps * dp_world_size
    val_samples = (
        (num_training_steps // eval_interval + 1)
        * eval_steps
        * micro_batch_size
        * gradient_accumulation_steps
        * dp_world_size
    )
    test_samples = eval_steps * micro_batch_size * gradient_accumulation_steps * dp_world_size

    return train_samples, val_samples, test_samples
