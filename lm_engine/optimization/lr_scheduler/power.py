# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from torch.optim import Optimizer

from .base import _LRScheduler
from .linear import _linear


def _power(a: float, b: float, x: float) -> float:
    return a * (x**b)


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
    ) -> PowerScheduler:
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
