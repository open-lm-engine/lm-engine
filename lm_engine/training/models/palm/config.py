# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Literal

from ...model_config import CommonConfig


class PaLMConfig(CommonConfig):
    model_type: Literal["palm"] = "palm"
