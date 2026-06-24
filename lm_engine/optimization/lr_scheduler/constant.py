# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from torch.optim import Optimizer

from .base import _LRScheduler
from .linear import _linear


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
    ) -> ConstantScheduler:
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
