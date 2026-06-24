# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ..arguments import BaseArgs
from ..modeling_utils import SoftPlusDecayGateArgs


class _Mamba2Args(SoftPlusDecayGateArgs):
    sequence_mixer_type: str = "mamba2"
    state_size: int
    intermediate_size: int
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


class _M2RNNArgs(SoftPlusDecayGateArgs):
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
