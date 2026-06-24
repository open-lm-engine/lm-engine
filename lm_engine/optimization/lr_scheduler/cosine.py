# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

from .base import _LRScheduler
from .linear import _linear


def _cosine(a: float, b: float, t: float, x: float) -> float:
    return a * (1 + math.cos(math.pi * x / t)) / 2 + b


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
