# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...accelerator import Accelerator, KernelBackend
from ...utils import is_triton_available
from .torch_implementation import _sgd_torch


_FUNCTIONS = {KernelBackend.torch: _sgd_torch}

if is_triton_available():
    from .triton_implementation import _sgd_triton

    _FUNCTIONS[KernelBackend.cuda] = _sgd_triton
    _FUNCTIONS[KernelBackend.triton] = _sgd_triton


@torch.no_grad()
def sgd(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    momentum_buffer_list: list[torch.Tensor | None],
    lr: float,
    weight_decay: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    step: int,
    *,
    kernel_backend: KernelBackend | None = None,
) -> None:
    if len(params) == 0:
        return

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    _FUNCTIONS[kernel_backend](
        params=params,
        grads=grads,
        momentum_buffer_list=momentum_buffer_list,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        step=step,
    )
