# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ...config import CommonConfig


class GPTBaseConfig(CommonConfig):
    model_type: str = "gpt_base"

    def model_post_init(self, __context: Any) -> None:
        assert self.model_type == "gpt_crosslayer"
