import logging

from ...arguments import TrainingArgs
from ...defaults import INPUT_FORMAT, OUTPUT_FORMAT
from ...tokenizers import TOKENIZER_TYPE
from ...utils import ProcessGroupManager, log_rank_0
from ..dataloader import ResumableDataLoader
from .blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from .blended_megatron_dataset_config import GPTDatasetConfig
from .gpt_dataset import GPTDataset
from .sampler import MegatronBatchSampler
from .utils import compile_helpers


def get_megatron_gpt_dataloaders(
    args: TrainingArgs, tokenizer: TOKENIZER_TYPE, consumed_samples: int
) -> tuple[ResumableDataLoader, list[ResumableDataLoader], list[ResumableDataLoader]]:
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

    # Option 1: data loading using --data-path with single file
    # Option 2: data loading using --data-path with multiple weighted files
    # Option 3: data loading using --(train|val|test)-data-path with multiple weighted files
    train_ds, val_ds, test_ds = gpt_dataset_builder.build()

    if not isinstance(val_ds, list):
        val_ds = [val_ds]
    if not isinstance(test_ds, list):
        test_ds = [test_ds]

    log_rank_0(logging.INFO, "> finished creating GPT datasets ...")

    def _get_dataloader(dataset: GPTDataset | None, consumed_samples: int):
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
) -> tuple[int, int, int]:
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
