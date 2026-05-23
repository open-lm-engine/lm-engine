# **************************************************
# Copyright (c) 2026, Mayank Mishra, Huanzhi Mao, Zhonglin Han
# **************************************************


import torch

from .cpp_extension import compile_cpp_extension
from .environment import environment
from .hf_hub import download_repo
from .miscellaneous import divide_if_divisible
from .mixed_precision import normalize_dtype_string, string_to_torch_dtype, torch_dtype_to_string
from .packages import (
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
    is_ray_available,
    is_sonicmoe_available,
    is_torch_neuronx_available,
    is_torch_xla_available,
    is_torchao_available,
    is_triton_available,
    is_wandb_available,
    is_xma_available,
    is_zstandard_available,
)
from .pydantic import BaseArgs
from .safetensors import SafeTensorsWeightsManager
from .wrapper import get_module_class_from_name
from .yaml import load_yaml


def setup_tf32(use_tf32: bool = True) -> None:
    """whether to use tf32 instead of fp32

    Args:
        use_tf32 (bool, optional): Defaults to True.
    """

    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
