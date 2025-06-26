# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


_STR_TO_TORCH_DTYPE = {
    # fp32
    "fp32": torch.float32,
    "float32": torch.float32,
    # fp16
    "fp16": torch.float16,
    "float16": torch.float16,
    # bf16
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

_TORCH_DTYPE_TO_STR = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


def string_to_torch_dtype(dtype_str: str) -> torch.dtype:
    assert dtype_str in _STR_TO_TORCH_DTYPE, f"{dtype_str} is not a supported torch dtype"
    return _STR_TO_TORCH_DTYPE[dtype_str]


def torch_dtype_to_string(torch_dtype: torch.dtype) -> str:
    assert torch_dtype in _TORCH_DTYPE_TO_STR, f"{torch_dtype} is not a supported torch dtype"
    return _TORCH_DTYPE_TO_STR[torch_dtype]


def normalize_dtype_string(dtype_str: str) -> str:
    if dtype_str in ["fp8", "float8"]:
        return "fp8"
    return torch_dtype_to_string(string_to_torch_dtype(dtype_str))
