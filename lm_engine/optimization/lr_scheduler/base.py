# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


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
    ) -> _LRScheduler:
        self.lr_warmup_boundary = num_warmup_steps
        self.lr_constant_boundary = self.lr_warmup_boundary + num_constant_steps

        self.lr_decay_boundary = num_training_steps
        if num_decay_steps is not None:
            self.lr_decay_boundary = self.lr_constant_boundary + num_decay_steps

        self.lr_decay_factor = lr_decay_factor

        super().__init__(optimizer, lr_lambda=self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(num_steps: int): ...
