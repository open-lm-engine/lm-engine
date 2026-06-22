# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from .base import _LRScheduler


def _linear(m: float, c: float, x: float) -> float:
    return m * x + c


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
