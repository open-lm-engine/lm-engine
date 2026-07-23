# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ...softplus_decay_gate import SoftPlusDecayGateArgs


class M2RNNArgs(SoftPlusDecayGateArgs):
    sequence_mixer_type: str = "m2rnn"
    k_head_dim: int
    v_head_dim: int
    num_q_heads: int
    num_k_heads: int
    num_v_heads: int
    num_f_heads: int
    num_g_heads: int
    num_weight_heads: int
    use_residual: bool
    kernel_size: int | None
    activation_function: str | None
    add_bias: bool
    gradient_clipping: float | None
    normalization_function: str | None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        assert self.sequence_mixer_type == "m2rnn"
