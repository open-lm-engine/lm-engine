# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging

import torch

from .accelerator import Accelerator
from .communication import Communication
from .environment import environment
from .hf_hub import download_repo
from .logger import log_environment, log_metrics, log_rank_0, print_rank_0, print_ranks_all, set_logger, warn_rank_0
from .loss_dict import MetricsTrackingDict
from .miscellaneous import divide_if_divisible
from .mixed_precision import normalize_dtype_string, string_to_torch_dtype, torch_dtype_to_string
from .packages import (
    is_aim_available,
    is_causal_conv1d_available,
    is_colorlog_available,
    is_fla_available,
    is_flash_attention_2_available,
    is_flash_attention_3_available,
    is_mamba_2_ssm_available,
    is_multi_storage_client_available,
    is_ray_available,
    is_sonicmoe_available,
    is_torch_xla_available,
    is_torchao_available,
    is_triton_available,
    is_wandb_available,
    is_xma_available,
    is_zstandard_available,
)
from .parallel import ProcessGroupManager, get_pipeline_stage_ids_on_current_rank, run_rank_n
from .profiler import TorchProfiler
from .pydantic import BaseArgs
from .random import set_seed
from .safetensors import SafeTensorsWeightsManager
from .step_tracker import StepTracker
from .tracking import ExperimentsTracker, ProgressBar
from .wrapper import get_module_class_from_name
from .yaml import load_yaml


def init_distributed(
    tensor_parallel_world_size: int,
    pipeline_parallel_world_size: int,
    data_parallel_replication_world_size: int,
    data_parallel_sharding_world_size: int,
    zero_stage: int,
    timeout_minutes: int = None,
    use_async_tensor_parallel: bool = False,
) -> None:
    """intialize distributed

    Args:
        tensor_parallel_world_size (int): tensor parallel size
        pipeline_parallel_world_size (int): pipeline parallel size
        data_parallel_replication_world_size (int): data parallel replication world size
        data_parallel_sharding_world_size (int): data parallel sharding world size
        zero_stage (int): zero stage
        timeout_minutes (int, optional): distributed timeout in minutes. Defaults to None.
        use_async_tensor_parallel (bool): whether to use async-TP. Defaults to False.
    """

    process_group_manager = ProcessGroupManager(
        tensor_parallel_world_size=tensor_parallel_world_size,
        pipeline_parallel_world_size=pipeline_parallel_world_size,
        data_parallel_replication_world_size=data_parallel_replication_world_size,
        data_parallel_sharding_world_size=data_parallel_sharding_world_size,
        zero_stage=zero_stage,
        timeout_minutes=timeout_minutes,
        use_async_tensor_parallel=use_async_tensor_parallel,
    )

    log_rank_0(logging.INFO, process_group_manager)
    log_rank_0(logging.INFO, f"total accelerators = {process_group_manager.get_world_size()}")
    log_rank_0(logging.INFO, f"tensor parallel size = {process_group_manager.get_tensor_parallel_world_size()}")
    log_rank_0(logging.INFO, f"pipeline parallel size = {process_group_manager.get_pipeline_parallel_world_size()}")
    log_rank_0(logging.INFO, f"data parallel size = {process_group_manager.get_data_parallel_world_size()}")

    for function, message in [
        (is_flash_attention_2_available, "Flash Attention 2 is not installed"),
        (is_flash_attention_3_available, "Flash Attention 3 is not installed"),
        (is_aim_available, "aim is not installed"),
        (is_wandb_available, "wandb is not installed"),
        (is_colorlog_available, "colorlog is not installed"),
        (is_triton_available, "OpenAI triton is not installed"),
        (
            is_xma_available,
            "accelerated-model-architectures is not installed, install from "
            "https://github.com/open-lm-engine/accelerated-model-architectures",
        ),
        (is_causal_conv1d_available, "causal-conv1d is not installed"),
        (is_mamba_2_ssm_available, "mamba-ssm is not installed"),
        (is_torchao_available, "torchao is not installed"),
        (is_zstandard_available, "zstandard is not available"),
        (is_torch_xla_available, "torch_xla is not available"),
        (is_multi_storage_client_available, "multi-storage-client is not available"),
    ]:
        if not function():
            warn_rank_0(message)


def setup_tf32(use_tf32: bool = True) -> None:
    """whether to use tf32 instead of fp32

    Args:
        use_tf32 (bool, optional): Defaults to True.
    """

    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
