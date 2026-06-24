# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

from .base import _LRScheduler
from .linear import _linear


def _exponential(a: float, b: float, t: float, x: float) -> float:
    return a * math.exp(-x / t) + b


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
