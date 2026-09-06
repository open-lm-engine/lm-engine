# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from ...accelerator import KernelBackend
from .op import sgd
from .param_group import SGDParamsGroup


class SGD:
    def __init__(self, param_groups: list[SGDParamsGroup]) -> SGD:
        self.param_groups = param_groups

    @torch.no_grad()
    def step(self, kernel_backend: KernelBackend | None = None) -> None:
        for group in self.param_groups:
            params, grads, momentum_buffer_list = group.get_params_for_optimization()
            step = group.increment_step()

            sgd(
                params=params,
                grads=grads,
                momentum_buffer_list=momentum_buffer_list,
                lr=group.lr,
                weight_decay=group.weight_decay,
                momentum=group.momentum,
                dampening=group.dampening,
                nesterov=group.nesterov,
                maximize=group.maximize,
                step=step,
                kernel_backend=kernel_backend,
            )
