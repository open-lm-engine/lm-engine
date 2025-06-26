# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..containers import BackwardHookOptimizerContainer, LRSchedulerContainer, ModelContainer, OptimizerContainer
from ..enums import LRDecaySchedule


def _linear(m: float, c: float, x: float) -> float:
    return m * x + c


def _cosine(a: float, b: float, t: float, x: float) -> float:
    return a * (1 + math.cos(math.pi * x / t)) / 2 + b


def _exponential(a: float, b: float, t: float, x: float) -> float:
    return a * math.exp(-x / t) + b


def _power(a: float, b: float, x: float) -> float:
    return a * (x**b)


class _LRScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_constant_steps: int,
        num_decay_steps: int,
        num_training_steps: int,
        lr_decay_factor: float,
        last_epoch: int = -1,
    ) -> None:
        self.lr_warmup_boundary = num_warmup_steps
        self.lr_constant_boundary = self.lr_warmup_boundary + num_constant_steps

        self.lr_decay_boundary = num_training_steps
        if num_decay_steps is not None:
            self.lr_decay_boundary = self.lr_constant_boundary + num_decay_steps

        self.lr_decay_factor = lr_decay_factor

        super().__init__(optimizer, lr_lambda=self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(num_steps: int): ...


class ConstantScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_constant_steps: int,
        num_decay_steps: int,
        num_training_steps: int,
        lr_decay_factor: float,
        last_epoch: int = -1,
    ) -> None:
        assert num_decay_steps == 0, "num_decay_steps should be 0 for constant schedule"

        super().__init__(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_constant_steps=num_constant_steps,
            num_decay_steps=num_decay_steps,
            num_training_steps=num_training_steps,
            lr_decay_factor=lr_decay_factor,
            last_epoch=last_epoch,
        )

    def _lr_lambda(self, num_steps: int) -> float:
        factor = (
            _linear(m=1 / self.lr_warmup_boundary, c=0, x=num_steps)
            if (self.lr_warmup_boundary > 0 and num_steps <= self.lr_warmup_boundary)
            else 1
        )
        return factor


class CosineScheduler(_LRScheduler):
    def _lr_lambda(self, num_steps: int) -> float:
        if self.lr_warmup_boundary > 0 and num_steps <= self.lr_warmup_boundary:
            factor = _linear(m=1 / self.lr_warmup_boundary, c=0, x=num_steps)
        elif num_steps <= self.lr_constant_boundary:
            factor = 1
        elif num_steps <= self.lr_decay_boundary:
            factor = _cosine(
                a=1 - self.lr_decay_factor,
                b=self.lr_decay_factor,
                t=self.lr_decay_boundary - self.lr_constant_boundary,
                x=num_steps - self.lr_constant_boundary,
            )
        else:
            factor = self.lr_decay_factor

        return factor


class ExponentialScheduler(_LRScheduler):
    def _lr_lambda(self, num_steps: int) -> float:
        if self.lr_warmup_boundary > 0 and num_steps <= self.lr_warmup_boundary:
            factor = _linear(m=1 / self.lr_warmup_boundary, c=0, x=num_steps)
        elif num_steps <= self.lr_constant_boundary:
            factor = 1
        # we use full decay phase for exponential unlike linear or cosine which have a constant phase in the end
        else:
            factor = _exponential(
                a=(1 - self.lr_decay_factor) * math.e / (math.e - 1),
                b=(self.lr_decay_factor * math.e - 1) / (math.e - 1),
                t=self.lr_decay_boundary - self.lr_constant_boundary,
                x=num_steps - self.lr_constant_boundary,
            )

        return factor


class LinearScheduler(_LRScheduler):
    def _lr_lambda(self, num_steps: int) -> float:
        if self.lr_warmup_boundary > 0 and num_steps <= self.lr_warmup_boundary:
            factor = _linear(m=1 / self.lr_warmup_boundary, c=0, x=num_steps)
        elif num_steps <= self.lr_constant_boundary:
            factor = 1
        elif num_steps <= self.lr_decay_boundary:
            factor = _linear(
                m=(self.lr_decay_factor - 1) / (self.lr_decay_boundary - self.lr_constant_boundary),
                c=1,
                x=num_steps - self.lr_constant_boundary,
            )
        else:
            factor = self.lr_decay_factor

        return factor


class PowerScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_constant_steps: int,
        num_decay_steps: int,
        num_training_steps: int,
        lr_decay_factor: float,
        a: float,
        b: float,
        c: float,
        last_epoch: int = -1,
    ) -> None:
        assert num_constant_steps == 0, "num_constant_steps should be 0 for power law scheduler"

        self.a = a
        self.b = b
        self.c = c

        self._optimizer_lr = optimizer.param_groups[0]["lr"]

        # cache max linear warmup y-axis value and avoid computing every time
        self._max_lr_during_warmup = min(
            1, _power(a=self.a / self._optimizer_lr, b=self.b, x=num_warmup_steps * self.c)
        )

        super().__init__(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_constant_steps=num_constant_steps,
            num_decay_steps=num_decay_steps,
            num_training_steps=num_training_steps,
            lr_decay_factor=lr_decay_factor,
            last_epoch=last_epoch,
        )

    def _lr_lambda(self, num_steps: int) -> float:
        if self.lr_warmup_boundary > 0 and num_steps <= self.lr_warmup_boundary:
            factor = _linear(m=self._max_lr_during_warmup / self.lr_warmup_boundary, c=0, x=num_steps)
        # note this might also include constant steps
        else:
            factor = min(1, _power(a=self.a / self._optimizer_lr, b=self.b, x=num_steps * self.c))

        return factor


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
