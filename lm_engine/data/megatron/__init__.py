import json
import logging
import os

from ...arguments import TrainingArgs
from ...defaults import INPUT_FORMAT, OUTPUT_FORMAT
from ...tokenizers import TOKENIZER_TYPE
from ...utils import Accelerator, ProcessGroupManager, log_rank_0
from ..dataloader import ResumableDataLoader
from .blended_megatron_dataset_builder import build
from .blended_megatron_dataset_config import GPTDatasetConfig
from .gpt_dataset import GPTDataset
from .sampler import MegatronBatchSampler
from .utils import compile_helpers


def _resolve_data_path(class_args: dict, key: str) -> list | None:
    """Resolve data_path from direct config or environment variable.
    
    Supports two ways to specify data paths:
    1. Direct config: `data_path: [weight, path, ...]`
    2. Environment variable name: `data_path_env: DATA_MIXTURE`
       The env var should contain a JSON array string, e.g.:
       export DATA_MIXTURE='[252, "msc://bucket/path/0", 252, "msc://bucket/path/1"]'
    """
    env_key = f"{key}_env"
    
    # Check for env var-based config first
    if class_args.get(env_key):
        env_var_name = class_args.get(env_key)
        env_value = os.environ.get(env_var_name)
        if not env_value:
            raise ValueError(f"Environment variable '{env_var_name}' not set or empty")
        return _parse_data_path_string(env_value)
    
    # Fall back to direct config
    return class_args.get(key)


def _parse_data_path_string(value: str) -> list:
    """Parse data path from JSON array string.
    
    Example: '[252, "msc://path/0", 252, "msc://path/1"]'
    """
    try:
        return json.loads(value.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in data path: {e}")


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

    compile_helpers()

    log_rank_0(logging.INFO, "> building train, validation, and test datasets for GPT ...")

    # Option 1: data loading using --data-path with single file
    # Option 2: data loading using --data-path with multiple weighted files
    # Option 3: data loading using --(train|val|test)-data-path with multiple weighted files
    # All options support external file via *_file suffix (e.g., data_path_file: /path/to/mixture.yml)
    train_ds, val_ds, test_ds = build(
        sizes=_get_train_val_test_samples(
            args.training_parameters.num_training_steps,
            micro_batch_size,
            gradient_accumulation_steps,
            args.training_parameters.eval_interval,
            class_args.get("eval_steps"),
        ),
        config=GPTDatasetConfig(
            sequence_length=class_args.get("sequence_length"),
            blend=_resolve_data_path(class_args, "data_path"),
            blend_per_split=[
                _resolve_data_path(class_args, "train_data_path"),
                _resolve_data_path(class_args, "val_data_path"),
                _resolve_data_path(class_args, "test_data_path"),
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

    if not isinstance(val_ds, list):
        val_ds = [val_ds]
    if not isinstance(test_ds, list):
        test_ds = [test_ds]

    log_rank_0(logging.INFO, "> finished creating GPT datasets ...")

    accelerator = Accelerator.get_accelerator()

    def _get_dataloader(dataset: GPTDataset | None, consumed_samples: int):
        if dataset is None:
            return None

        dataloader = ResumableDataLoader(
            dataset,
            batch_sampler=MegatronBatchSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=(
                    micro_batch_size if num_pipeline_stages == 1 else micro_batch_size * gradient_accumulation_steps
                ),
                num_replicas=ProcessGroupManager.get_data_parallel_world_size(),
                rank=ProcessGroupManager.get_data_parallel_rank(),
            ),
            multiprocessing_context="fork" if accelerator == Accelerator.tpu else None,
            num_workers=0 if accelerator == Accelerator.trainium else class_args.get("num_workers", 2),
            pin_memory=accelerator != Accelerator.trainium,
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
