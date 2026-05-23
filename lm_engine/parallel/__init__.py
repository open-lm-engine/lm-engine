# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from .manager import ProcessGroupManager, run_rank_n


def init_distributed(
    tensor_parallel_world_size: int,
    pipeline_parallel_world_size: int,
    data_parallel_replication_world_size: int,
    data_parallel_sharding_world_size: int,
    context_parallel_world_size: int,
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

    import logging

    from ..logging_utils import log_environment, log_rank_0, warn_rank_0
    from ..utils import (
        is_aim_available,
        is_causal_conv1d_available,
        is_colorlog_available,
        is_flash_attention_2_available,
        is_flash_attention_3_available,
        is_flash_attention_4_available,
        is_mamba_2_ssm_available,
        is_multi_storage_client_available,
        is_quack_available,
        is_torch_xla_available,
        is_torchao_available,
        is_triton_available,
        is_wandb_available,
        is_xma_available,
        is_zstandard_available,
    )

    process_group_manager = ProcessGroupManager(
        tensor_parallel_world_size=tensor_parallel_world_size,
        pipeline_parallel_world_size=pipeline_parallel_world_size,
        data_parallel_replication_world_size=data_parallel_replication_world_size,
        data_parallel_sharding_world_size=data_parallel_sharding_world_size,
        context_parallel_world_size=context_parallel_world_size,
        zero_stage=zero_stage,
        timeout_minutes=timeout_minutes,
        use_async_tensor_parallel=use_async_tensor_parallel,
    )

    log_rank_0(logging.INFO, process_group_manager)
    log_rank_0(logging.INFO, f"total accelerators = {process_group_manager.get_world_size()}")
    log_rank_0(logging.INFO, f"tensor parallel size = {process_group_manager.get_tensor_parallel_world_size()}")
    log_rank_0(logging.INFO, f"pipeline parallel size = {process_group_manager.get_pipeline_parallel_world_size()}")
    log_rank_0(logging.INFO, f"data parallel size = {process_group_manager.get_data_parallel_world_size()}")
    log_rank_0(logging.INFO, f"context parallel size = {process_group_manager.get_context_parallel_world_size()}")

    for function, message in [
        (is_flash_attention_2_available, "Flash Attention 2 is not installed"),
        (is_flash_attention_3_available, "Flash Attention 3 is not installed"),
        (is_flash_attention_4_available, "Flash Attention 4 is not installed"),
        (is_aim_available, "aim is not installed"),
        (is_wandb_available, "wandb is not installed"),
        (is_colorlog_available, "colorlog is not installed"),
        (is_triton_available, "OpenAI triton is not installed"),
        (
            is_xma_available,
            "accelerated-model-architectures is not installed, install lm-engine with the xma extra",
        ),
        (is_causal_conv1d_available, "causal-conv1d is not installed"),
        (is_mamba_2_ssm_available, "mamba-ssm is not installed"),
        (is_torchao_available, "torchao is not installed"),
        (is_zstandard_available, "zstandard is not available"),
        (is_torch_xla_available, "torch_xla is not available"),
        (is_multi_storage_client_available, "multi-storage-client is not available"),
        (is_quack_available, "quack-kernels is not installed"),
    ]:
        if not function():
            warn_rank_0(message)
