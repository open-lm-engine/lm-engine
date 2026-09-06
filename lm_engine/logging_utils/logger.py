# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import logging
from importlib.metadata import distributions
from warnings import warn

from ..parallel import ProcessGroupManager, run_rank_n
from ..utils import (
    is_aim_available,
    is_causal_conv1d_available,
    is_colorlog_available,
    is_fla_available,
    is_flash_attention_2_available,
    is_flash_attention_3_available,
    is_flash_attention_4_available,
    is_mamba_2_ssm_available,
    is_multi_storage_client_available,
    is_quack_available,
    is_sonicmoe_available,
    is_torch_neuronx_available,
    is_torch_xla_available,
    is_torchao_available,
    is_triton_available,
    is_wandb_available,
    is_xma_available,
    is_zstandard_available,
)
from .tracking import is_tracking_rank


if is_colorlog_available():
    from colorlog import ColoredFormatter


_LOGGER: logging.Logger = None


def set_logger(level: int = logging.INFO, colored_log: bool = False) -> None:
    stream = logging.StreamHandler()

    if colored_log:
        stream.setFormatter(ColoredFormatter("%(asctime)s - %(log_color)s[%(levelname)-8s] ▶%(reset)s %(message)s"))
        logging.basicConfig(level=level, handlers=[stream], force=True)
    else:
        logging.basicConfig(
            level=level, handlers=[stream], format="%(asctime)s - [%(levelname)-8s] ▶ %(message)s", force=True
        )

    global _LOGGER
    _LOGGER = logging.getLogger()


def get_logger() -> logging.Logger:
    return _LOGGER


@run_rank_n
def log_rank_0(level: int, msg: str) -> None:
    logger = get_logger()

    if logger is None:
        set_logger()
        log_rank_0(logging.WARN, "logger is not initialized yet, initializing now")
    else:
        logger.log(level=level, msg=msg, stacklevel=3)


def log_metrics(level: int, msg: str) -> None:
    if not is_tracking_rank():
        return

    get_logger().log(level=level, msg=msg, stacklevel=3)


@run_rank_n
def print_rank_0(*args, **kwargs) -> None:
    """print on a single process"""

    print(*args, **kwargs)


def print_ranks_all(*args, **kwargs) -> None:
    """print on all processes sequentially, blocks other process and is slow. Please us sparingly."""

    for rank in range(ProcessGroupManager.get_world_size()):
        run_rank_n(print, rank=rank, barrier=True)(f"rank {rank}:", *args, **kwargs)


@run_rank_n
def warn_rank_0(*args, **kwargs) -> None:
    """warn on a single process"""

    warn(*args, **kwargs, stacklevel=3)


@run_rank_n
def log_environment() -> None:
    packages = sorted(["{}=={}".format(d.metadata["Name"], d.version) for d in distributions()])

    for function, message in [
        (is_aim_available, "aim is not installed"),
        (is_causal_conv1d_available, "causal-conv1d is not installed"),
        (is_colorlog_available, "colorlog is not installed"),
        (is_fla_available, "fla is not installed"),
        (is_flash_attention_2_available, "Flash Attention 2 is not installed"),
        (is_flash_attention_3_available, "Flash Attention 3 is not installed"),
        (is_flash_attention_4_available, "Flash Attention 4 is not installed"),
        (is_mamba_2_ssm_available, "mamba-ssm is not installed"),
        (is_multi_storage_client_available, "multi-storage-client is not available"),
        (is_quack_available, "quack-kernels is not installed"),
        (is_sonicmoe_available, "sonicmoe is not installed"),
        (is_torch_xla_available, "torch_xla is not available"),
        (is_torchao_available, "torchao is not installed"),
        (is_torch_neuronx_available, "torch-neuronx is not installed"),
        (is_triton_available, "OpenAI triton is not installed"),
        (is_wandb_available, "wandb is not installed"),
        (
            is_xma_available,
            "accelerated-model-architectures is not installed, install lm-engine with the xma extra",
        ),
        (is_zstandard_available, "zstandard is not available"),
    ]:
        if not function():
            warn_rank_0(message)

    log_rank_0(logging.INFO, "------------------------ packages ------------------------")
    for package in packages:
        log_rank_0(logging.INFO, package)
    log_rank_0(logging.INFO, "-------------------- end of packages ---------------------")
