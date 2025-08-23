# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


torch.mps.is_initialized = lambda: True
torch.mps.current_device = lambda: 0


def get_backend() -> str:
    if torch.cuda.is_available():
        backend = "cpu:gloo,cuda:nccl"
    elif torch.mps.is_available():
        backend = "cpu:gloo"
    else:
        raise NotImplementedError()

    return backend


def get_device_string() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        raise NotImplementedError()

    return device


def set_device(device: torch.device) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(device)


def get_current_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    elif torch.mps.is_available():
        device = torch.mps.current_device()
    else:
        raise NotImplementedError()

    return device
