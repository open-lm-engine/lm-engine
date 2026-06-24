# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from ...softplus_decay_gate import SoftPlusDecayGateArgs


class GatedDeltaNetArgs(SoftPlusDecayGateArgs):
    sequence_mixer_type: str = "gated_deltanet"
    k_head_dim: int
    v_head_dim: int
    num_k_heads: int
    num_v_heads: int
    use_gate: bool
    attention_multiplier: float | None
    allow_neg_eigval: bool
    kernel_size: int

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        assert self.sequence_mixer_type == "gated_deltanet"
