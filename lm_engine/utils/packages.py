# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


try:
    import flash_attn

    _IS_FLASH_ATTENTION_2_AVAILABLE = True
except ImportError:
    _IS_FLASH_ATTENTION_2_AVAILABLE = False


def is_flash_attention_2_available() -> bool:
    return _IS_FLASH_ATTENTION_2_AVAILABLE


try:
    from flash_attn_interface import flash_attn_func

    _IS_FLASH_ATTENTION_3_AVAILABLE = True
except ImportError:
    _IS_FLASH_ATTENTION_3_AVAILABLE = False


def is_flash_attention_3_available() -> bool:
    return _IS_FLASH_ATTENTION_3_AVAILABLE


try:
    import aim

    _IS_AIM_AVAILABLE = True
except ImportError:
    _IS_AIM_AVAILABLE = False


def is_aim_available() -> bool:
    return _IS_AIM_AVAILABLE


try:
    import wandb

    _IS_WANDB_AVAILABLE = True
except ImportError:
    _IS_WANDB_AVAILABLE = False


def is_wandb_available() -> bool:
    return _IS_WANDB_AVAILABLE


try:
    import colorlog

    _IS_COLORLOG_AVAILABLE = True
except ImportError:
    _IS_COLORLOG_AVAILABLE = False


def is_colorlog_available() -> bool:
    return _IS_COLORLOG_AVAILABLE


try:
    import triton

    _IS_TRITON_AVAILABLE = True
except ImportError:
    _IS_TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _IS_TRITON_AVAILABLE


try:
    import xma

    _IS_XMA_AVAILABLE = True
except:
    _IS_XMA_AVAILABLE = False


def is_xma_available() -> bool:
    return _IS_XMA_AVAILABLE


try:
    import causal_conv1d

    _IS_CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    _IS_CAUSAL_CONV1D_AVAILABLE = False


def is_causal_conv1d_available() -> bool:
    return _IS_CAUSAL_CONV1D_AVAILABLE


try:
    import mamba_ssm

    _IS_MAMBA_2_SSM_AVAILABLE = True
except ImportError:
    _IS_MAMBA_2_SSM_AVAILABLE = False


def is_mamba_2_ssm_available() -> bool:
    return _IS_MAMBA_2_SSM_AVAILABLE


try:
    import torchao

    _IS_TORCHAO_AVAILABLE = True
except ImportError:
    _IS_TORCHAO_AVAILABLE = False


def is_torchao_available() -> bool:
    return _IS_TORCHAO_AVAILABLE


try:
    import zstandard

    _IS_ZSTANDARD_AVAILABLE = True
except ImportError:
    _IS_ZSTANDARD_AVAILABLE = False


def is_zstandard_available() -> bool:
    return _IS_ZSTANDARD_AVAILABLE


try:
    import torch_xla

    _IS_TORCH_XLA_AVAILABLE = True
except ImportError:
    _IS_TORCH_XLA_AVAILABLE = False


def is_torch_xla_available() -> bool:
    return _IS_TORCH_XLA_AVAILABLE


try:
    import torch_neuronx

    _IS_TORCH_NEURONX_AVAILABLE = True
except ImportError:
    _IS_TORCH_NEURONX_AVAILABLE = False


def is_torch_neuronx_available() -> bool:
    return _IS_TORCH_NEURONX_AVAILABLE


try:
    import multistorageclient

    _IS_MULTI_STORAGE_CLIENT_AVAILABLE = True
except ImportError:
    _IS_MULTI_STORAGE_CLIENT_AVAILABLE = False


def is_multi_storage_client_available() -> bool:
    return _IS_MULTI_STORAGE_CLIENT_AVAILABLE


try:
    import sonicmoe

    _IS_SONIC_MOE_AVAILABLE = True
except ImportError:
    _IS_SONIC_MOE_AVAILABLE = False


def is_sonicmoe_available() -> bool:
    return _IS_SONIC_MOE_AVAILABLE


try:
    import ray

    _IS_RAY_AVAILABLE = True
except ImportError:
    _IS_RAY_AVAILABLE = False


def is_ray_available() -> bool:
    return _IS_RAY_AVAILABLE


try:
    import fla

    _IS_FLA_AVAILABLE = True
except ImportError:
    _IS_FLA_AVAILABLE = False


def is_fla_available() -> bool:
    return _IS_FLA_AVAILABLE
