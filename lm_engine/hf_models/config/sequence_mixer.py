# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Any

from ...utils import BaseArgs


class _SoftmaxAttentionArgs(BaseArgs):
    sequence_mixer_type: str = "softmax_attention"
    num_attention_heads: int = 12
    num_key_value_heads: int = 1
    softmax_dropout: float = 0
    dropout: float = 0
    add_bias: bool = False
    attention_multiplier: float | None = None
    sliding_window: int | None = None
    # needed for Qwen 2 MoE
    qkv_bias: bool = None

    def model_post_init(self, __context: Any) -> None:
        if self.qkv_bias is None:
            self.qkv_bias = self.add_bias

        assert self.sequence_mixer_type == "softmax_attention"


class _MultiHeadLatentAttentionArgs(BaseArgs):
    sequence_mixer_type: str = "multihead_latent_attention"
    num_attention_heads: int | None = None
    softmax_dropout: float = 0
    dropout: float = 0
    add_bias: bool = False
    attention_multiplier: float | None = None
    sliding_window: int | None = None
    query_compression_size: int | None = None
    key_value_compression_size: int | None = None
    num_attention_heads: int | None = None
    head_dim: int | None = None
    normalization_function: str = "layernorm"

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "multihead_latent_attention"
        assert self.num_attention_heads is not None
        assert self.query_compression_size is not None
        assert self.key_value_compression_size is not None
        assert self.num_attention_heads is not None
        assert self.head_dim is not None


class _GRUArgs(BaseArgs):
    sequence_mixer_type: str = "gru"
    state_head_dim: int
    num_input_heads: int
    num_forget_input_heads: int
    num_reset_input_heads: int
    num_weight_heads: int
    num_forget_weight_heads: int
    num_reset_weight_heads: int
    add_bias: bool = False
    normalization_function: str | None = None
    gradient_clipping: float | None = None
    kernel_size: int | None = None
    activation_function: str | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "gru"


class _RNNArgs(BaseArgs):
    sequence_mixer_type: str = "rnn"
    state_head_dim: int
    num_input_heads: int
    num_weight_heads: int
    add_bias: bool = False
    normalization_function: str | None = None
    gradient_clipping: float | None = None
    kernel_size: int | None = None
    activation_function: str | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "rnn"


class _CausalConvolution(BaseArgs):
    sequence_mixer_type: str = "causal_convolution"
    activation_function: str = "silu"
    in_channels: int
    out_channels: int
    kernel_size: int
    num_groups: int
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "causal_convolution"


class _SoftPlusDecayArgs(BaseArgs):
    A_init_min: float = 0
    A_init_max: float = 16
    dt_init_min: float = 0.001
    dt_init_max: float = 0.1
    dt_init_floor: float = 1e-4

    def model_post_init(self, __context: Any) -> None:
        assert self.A_init_min >= 0
        assert self.A_init_min <= self.A_init_max
        assert self.dt_min <= self.dt_max


class _GatedDeltaNetArgs(_SoftPlusDecayArgs):
    sequence_mixer_type: str = "gated_deltanet"
    k_head_dim: int
    v_head_dim: int
    num_k_heads: int
    num_v_heads: int
    use_gate: bool
    attention_multiplier: float | None = None
    allow_neg_eigval: bool
    kernel_size: int

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "gated_deltanet"


class _Mamba2Args(_SoftPlusDecayArgs):
    sequence_mixer_type: str = "mamba2"
    state_size: int = 128
    intermediate_size: int
    num_heads: int = 128
    conv_kernel_size: int = 4
    time_step_limit: tuple[float, float] = (0, float("inf"))
    add_bias: bool = False
    use_conv_bias: bool = True
    activation_function: str = "silu"
    num_groups: int = 8
    chunk_size: int = 256
    normalization_function: str | None = "rmsnorm"

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "mamba2"
