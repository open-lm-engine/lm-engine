# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ....arguments import BaseArgs


class RNNArgs(BaseArgs):
    sequence_mixer_type: str = "rnn"
    state_head_dim: int
    num_input_heads: int
    num_weight_heads: int
    add_bias: bool
    normalization_function: str | None
    gradient_clipping: float | None
    kernel_size: int | None
    activation_function: str | None

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "rnn"
