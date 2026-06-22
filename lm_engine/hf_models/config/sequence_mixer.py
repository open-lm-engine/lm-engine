# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ...arguments import BaseArgs


ATTENTION_MULTIPLIER_INVERSE_SQRT_METHOD = "1 / sqrt(head_dim)"
ATTENTION_MULTIPLIER_INVERSE_METHOD = "1 / head_dim"


class _SoftmaxAttentionArgs(BaseArgs):
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


class _SoftPlusDecayArgs(BaseArgs):
    A_init_min: float = 0
    A_init_max: float = 16
    dt_init_min: float = 0.001
    dt_init_max: float = 0.1
    dt_init_floor: float = 1e-4

    def model_post_init(self, __context: Any) -> None:
        assert self.A_init_min >= 0
        assert self.A_init_min <= self.A_init_max
        assert self.dt_init_min <= self.dt_init_max


class _Mamba2Args(_SoftPlusDecayArgs):
    sequence_mixer_type: str = "mamba2"
    state_size: int
    intermediate_size: int | None
    num_heads: int
    conv_kernel_size: int
    time_step_limit: tuple[float, float] = (0, float("inf"))
    add_bias: bool = False
    use_conv_bias: bool = True
    activation_function: str
    num_groups: int
    chunk_size: int = 256
    normalization_function: str | None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        assert self.sequence_mixer_type == "mamba2"


class _GRUArgs(BaseArgs):
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


class _RNNArgs(BaseArgs):
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


class _M2RNNArgs(_SoftPlusDecayArgs):
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


class _GatedDeltaNetArgs(_SoftPlusDecayArgs):
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
