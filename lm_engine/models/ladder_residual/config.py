# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Literal

from ...model_config import CommonConfig


class LadderResidualConfig(CommonConfig):
    model_type: Literal["ladder_residual"] = "ladder_residual"
