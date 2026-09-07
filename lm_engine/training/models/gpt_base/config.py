# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Literal

from ...model_config import CommonConfig


class GPTBaseConfig(CommonConfig):
    model_type: Literal["gpt_base"] = "gpt_base"
