# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Any

from ....arguments import BaseArgs


ATTENTION_MULTIPLIER_INVERSE_SQRT_METHOD = "1 / sqrt(head_dim)"
ATTENTION_MULTIPLIER_INVERSE_METHOD = "1 / head_dim"


class SoftmaxAttentionArgs(BaseArgs):
    sequence_mixer_type: str = "softmax_attention"
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int | None = None
    softmax_dropout: float = 0
    dropout: float = 0
    add_bias: bool = False
    attention_multiplier: float | None = None
    attention_multiplier_method: str | None = ATTENTION_MULTIPLIER_INVERSE_SQRT_METHOD
    attention_gate: bool = False
    exclusive_self_attention: bool = False
    sliding_window: int | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.attention_multiplier_method in [
            ATTENTION_MULTIPLIER_INVERSE_SQRT_METHOD,
            ATTENTION_MULTIPLIER_INVERSE_METHOD,
            None,
        ]

        if self.attention_multiplier_method in [
            ATTENTION_MULTIPLIER_INVERSE_SQRT_METHOD,
            ATTENTION_MULTIPLIER_INVERSE_METHOD,
        ]:
            assert self.attention_multiplier is None
        else:
            assert self.attention_multiplier is not None

        assert self.sequence_mixer_type == "softmax_attention"
