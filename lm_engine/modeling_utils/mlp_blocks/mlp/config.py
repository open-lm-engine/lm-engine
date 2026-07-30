# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any, Literal

from ....arguments import BaseArgs


class MLPArgs(BaseArgs):
    mlp_type: Literal["MLP"] = "MLP"
    intermediate_size: int
    activation_function: str
    dropout: float = 0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MLP"
