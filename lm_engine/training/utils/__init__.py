# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************


import torch

from .generation_cache import disable_generation_cache, is_generation_cache_enabled
from .hf_hub import download_repo
from .mixed_precision import normalize_dtype_string, string_to_torch_dtype, torch_dtype_to_string
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
