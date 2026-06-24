# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ...model_config import CommonConfig


class PaLMConfig(CommonConfig):
    model_type: str = "palm"

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        assert self.model_type == "palm"
