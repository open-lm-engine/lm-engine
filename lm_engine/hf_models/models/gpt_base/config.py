# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ....model_config import CommonConfig


class GPTBaseConfig(CommonConfig):
    model_type: str = "gpt_base"

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        assert self.model_type == "gpt_base"
