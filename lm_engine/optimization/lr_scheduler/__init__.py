# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from ...containers import BackwardHookOptimizerContainer, LRSchedulerContainer, ModelContainer, OptimizerContainer
from ...enums import LRDecaySchedule
from .constant import ConstantScheduler
from .cosine import CosineScheduler
from .exponential import ExponentialScheduler
from .linear import LinearScheduler
from .power import PowerScheduler


_LR_SCHEDULER_CLASSES = {
    LRDecaySchedule.constant: ConstantScheduler,
    LRDecaySchedule.linear: LinearScheduler,
    LRDecaySchedule.exponential: ExponentialScheduler,
    LRDecaySchedule.cosine: CosineScheduler,
    LRDecaySchedule.power: PowerScheduler,
}


def get_scheduler_container(
    model_container: ModelContainer,
    optimizer_container: OptimizerContainer,
    num_warmup_steps: int,
    num_constant_steps: int,
    num_decay_steps: int,
    num_training_steps: int,
    lr_decay_style: LRDecaySchedule,
    lr_decay_factor: float,
    extra_lr_scheduler_args: dict,
    use_optimizer_with_backward_hook: bool,
    last_epoch: int = -1,
) -> LambdaLR:
    if lr_decay_style not in _LR_SCHEDULER_CLASSES:
        raise ValueError(f"invalid lr_decay_style ({lr_decay_style})")

    if use_optimizer_with_backward_hook:
        for model in model_container:
            for param in model.parameters():
                param._lr_scheduler = _LR_SCHEDULER_CLASSES[lr_decay_style](
                    param._optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_constant_steps=num_constant_steps,
                    num_decay_steps=num_decay_steps,
                    num_training_steps=num_training_steps,
                    lr_decay_factor=lr_decay_factor,
                    **extra_lr_scheduler_args,
                    last_epoch=last_epoch,
                )

                def _step(p: nn.Parameter) -> None:
                    p._lr_scheduler.step()

                param.register_post_accumulate_grad_hook(_step)

        lr_scheduler_list = BackwardHookOptimizerContainer([None] * len(model_container))
    else:
        lr_scheduler_list = [
            _LR_SCHEDULER_CLASSES[lr_decay_style](
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_constant_steps=num_constant_steps,
                num_decay_steps=num_decay_steps,
                num_training_steps=num_training_steps,
                lr_decay_factor=lr_decay_factor,
                **extra_lr_scheduler_args,
                last_epoch=last_epoch,
            )
            for optimizer in optimizer_container
        ]

        lr_scheduler_list = LRSchedulerContainer(lr_scheduler_list)

    return lr_scheduler_list
