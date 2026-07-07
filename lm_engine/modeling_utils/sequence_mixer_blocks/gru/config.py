# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ....arguments import BaseArgs


class GRUArgs(BaseArgs):
    sequence_mixer_type: str = "gru"
    state_head_dim: int
    num_input_heads: int
    num_forget_input_heads: int
    num_reset_input_heads: int
    num_weight_heads: int
    num_forget_weight_heads: int
    num_reset_weight_heads: int
    add_bias: bool
    normalization_function: str | None
    gradient_clipping: float | None
    kernel_size: int | None
    activation_function: str | None

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "gru"
