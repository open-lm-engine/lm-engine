# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os

import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..arguments import TrainingArgs
from ..containers import LRSchedulerContainer, ModelContainer, OptimizerContainer
from ..optimization import get_scheduler_container
from ..utils import ProcessGroupManager


class _LRSchedulerSaver(Stateful):
    def __init__(self, lr_scheduler_container: LRSchedulerContainer) -> None:
        self.lr_scheduler_container = lr_scheduler_container

    def state_dict(self) -> dict:
        return [lr_scheduler.state_dict() for lr_scheduler in self.lr_scheduler_container]

    def load_state_dict(self, state_dict: list[dict]) -> None:
        assert len(self.lr_scheduler_container) == len(state_dict)

        for lr_scheduler, lr_scheduler_state_dict in zip(self.lr_scheduler_container, state_dict):
            lr_scheduler.load_state_dict(lr_scheduler_state_dict)


def _resume_learning_rate(
    args: TrainingArgs, model: nn.Module, optimizer: Optimizer, lr_scheduler: LambdaLR, iteration: int | None = None
) -> None:
    initial_lr = []
    for grp in optimizer.param_groups:
        initial_lr.append(grp["initial_lr"])
        grp["initial_lr"] = grp["lr"]

    # we create lr scheduler again here since optimizer is loaded from disk and lr scheduler is now out of sync
    # this helps to resume phase 2
    lr_scheduler_tmp = get_scheduler_container(
        model_container=ModelContainer([model]),
        optimizer_container=OptimizerContainer([optimizer]),
        num_warmup_steps=args.lr_scheduler_args.num_warmup_steps,
        num_constant_steps=args.lr_scheduler_args.num_constant_steps,
        num_decay_steps=args.lr_scheduler_args.num_decay_steps,
        num_training_steps=args.training_parameters.num_training_steps,
        lr_decay_style=args.lr_scheduler_args.lr_decay_style,
        lr_decay_factor=args.lr_scheduler_args.lr_decay_factor,
        extra_lr_scheduler_args=args.lr_scheduler_args.extra_lr_scheduler_args,
        use_optimizer_with_backward_hook=args.optimizer_args.use_optimizer_with_backward_hook,
        last_epoch=-1 if iteration is None else iteration - 1,
    )[0]

    for grp, lr_ in zip(optimizer.param_groups, initial_lr):
        grp["initial_lr"] = lr_

    lr_scheduler.load_state_dict(lr_scheduler_tmp.state_dict())
    del lr_scheduler_tmp


def _get_lr_scheduler_path(path: str) -> str:
    return os.path.join(path, "lr_scheduler", f"lr_scheduler-{ProcessGroupManager.get_global_rank()}.pt")
