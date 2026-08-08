# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Literal

from ...softplus_decay_gate import SoftPlusDecayGateArgs


class GatedDeltaNetArgs(SoftPlusDecayGateArgs):
    sequence_mixer_type: Literal["gated_deltanet"] = "gated_deltanet"
    k_head_dim: int
    v_head_dim: int
    num_k_heads: int
    num_v_heads: int
    use_gate: bool
    attention_multiplier: float | None
    allow_neg_eigval: bool
    kernel_size: int
