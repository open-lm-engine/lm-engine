# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ...config import CommonConfig


class LadderResidualConfig(CommonConfig):
    model_type: str = "ladder_residual"

    def model_post_init(self, __context: Any) -> None:
        assert self.model_type == "ladder_residual"
