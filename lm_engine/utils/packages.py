# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging
from importlib.metadata import distributions

import torch

from .logger import log_rank_0, warn_rank_0
from .parallel import run_rank_n


try:
    import flash_attn

    _IS_FLASH_ATTENTION_2_AVAILABLE = True
except ImportError:
    _IS_FLASH_ATTENTION_2_AVAILABLE = False

    warn_rank_0("Flash Attention 2 is not installed")


def is_flash_attention_2_available() -> bool:
    return _IS_FLASH_ATTENTION_2_AVAILABLE


try:
    import flash_attn_3_cuda

    _IS_FLASH_ATTENTION_3_AVAILABLE = True
except ImportError:
    _IS_FLASH_ATTENTION_3_AVAILABLE = False

    warn_rank_0("Flash Attention 3 is not installed")


def is_flash_attention_3_available() -> bool:
    return _IS_FLASH_ATTENTION_3_AVAILABLE


try:
    import aim

    _IS_AIM_AVAILABLE = True
except ImportError:
    _IS_AIM_AVAILABLE = False

    warn_rank_0("aim is not installed")


def is_aim_available() -> bool:
    return _IS_AIM_AVAILABLE


try:
    import wandb

    _IS_WANDB_AVAILABLE = True
except ImportError:
    _IS_WANDB_AVAILABLE = False

    warn_rank_0("wandb is not installed")


def is_wandb_available() -> bool:
    return _IS_WANDB_AVAILABLE


try:
    import colorlog

    _IS_COLORLOG_AVAILABLE = True
except ImportError:
    _IS_COLORLOG_AVAILABLE = False

    warn_rank_0("colorlog is not installed")


def is_colorlog_available() -> bool:
    return _IS_COLORLOG_AVAILABLE


try:
    import triton

    _IS_TRITON_AVAILABLE = True
except ImportError:
    _IS_TRITON_AVAILABLE = False

    warn_rank_0("OpenAI triton is not installed")


def is_triton_available() -> bool:
    return _IS_TRITON_AVAILABLE


try:
    if torch.cuda.is_available():
        import cute_kernels

        _IS_CUTE_KERNELS_AVAILABLE = True
    else:
        _IS_CUTE_KERNELS_AVAILABLE = False
except ImportError:
    _IS_CUTE_KERNELS_AVAILABLE = False

    warn_rank_0("cute-kernels is not installed, install from https://github.com/mayank31398/cute-kernels")


def is_cute_kernels_available() -> bool:
    return _IS_CUTE_KERNELS_AVAILABLE


try:
    if torch.cuda.is_available():
        import causal_conv1d

        _IS_CAUSAL_CONV1D_AVAILABLE = True
    else:
        _IS_CAUSAL_CONV1D_AVAILABLE = False
except ImportError:
    _IS_CAUSAL_CONV1D_AVAILABLE = False

    warn_rank_0("causal-conv1d is not installed")


def is_causal_conv1d_available() -> bool:
    return _IS_CAUSAL_CONV1D_AVAILABLE


try:
    if torch.cuda.is_available():
        import mamba_ssm

        _IS_MAMBA_2_SSM_AVAILABLE = True
    else:
        _IS_MAMBA_2_SSM_AVAILABLE = False
except ImportError:
    _IS_MAMBA_2_SSM_AVAILABLE = False

    warn_rank_0("mamba-ssm is not installed")


def is_mamba_2_ssm_available() -> bool:
    return _IS_MAMBA_2_SSM_AVAILABLE


try:
    import torchao

    _IS_TORCHAO_AVAILABLE = True
except ImportError:
    _IS_TORCHAO_AVAILABLE = False

    warn_rank_0("torchao is not installed")


def is_torchao_available() -> bool:
    return _IS_TORCHAO_AVAILABLE


try:
    import stickbreaking_attention

    _IS_STICKBREAKING_AVAILABLE = True
except ImportError:
    _IS_STICKBREAKING_AVAILABLE = False

    warn_rank_0(
        "stickbreaking-attention is not available, install from https://github.com/shawntan/stickbreaking-attention"
    )


def is_stickbreaking_available():
    return _IS_STICKBREAKING_AVAILABLE


try:
    import zstandard

    _IS_ZSTANDARD_AVAILABLE = True
except ImportError:
    _IS_ZSTANDARD_AVAILABLE = False

    warn_rank_0("zstandard is not available")


def is_zstandard_available():
    return _IS_STICKBREAKING_AVAILABLE


@run_rank_n
def log_environment() -> None:
    packages = sorted(["{}=={}".format(d.metadata["Name"], d.version) for d in distributions()])

    log_rank_0(logging.INFO, "------------------------ packages ------------------------")
    for package in packages:
        log_rank_0(logging.INFO, package)
    log_rank_0(logging.INFO, "-------------------- end of packages ---------------------")
