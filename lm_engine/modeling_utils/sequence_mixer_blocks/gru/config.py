# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Literal

from ....arguments import BaseArgs


class GRUArgs(BaseArgs):
    sequence_mixer_type: Literal["gru"] = "gru"
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
