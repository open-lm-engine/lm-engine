# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ...arguments import BaseArgs


class SoftPlusDecayGateArgs(BaseArgs):
    A_init_min: float = 0
    A_init_max: float = 16
    dt_init_min: float = 0.001
    dt_init_max: float = 0.1
    dt_init_floor: float = 1e-4

    def model_post_init(self, __context: Any) -> None:
        assert self.A_init_min >= 0
        assert self.A_init_min <= self.A_init_max
        assert self.dt_init_min <= self.dt_init_max
