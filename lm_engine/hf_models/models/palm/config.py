# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ...config import CommonConfig


class PaLMConfig(CommonConfig):
    model_type: str = "palm"

    def model_post_init(self, __context: Any) -> None:
        assert self.model_type == "palm"
