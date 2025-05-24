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
    add_bias: bool = True
    attention_multiplier: float | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "softmax_attention"


class _MultiHeadLatentAttentionArgs(BaseArgs):
    sequence_mixer_type: str = "multihead_latent_attention"
    num_attention_heads: int | None = None
    softmax_dropout: float = 0
    dropout: float = 0
    add_bias: bool = True
    attention_multiplier: float | None = None
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


class _StickbreakingAttentionArgs(BaseArgs):
    sequence_mixer_type: str = "stickbreaking_attention"
    num_attention_heads: int = 12
    num_key_value_heads: int = 1
    dropout: float = 0
    add_bias: bool = True
    attention_multiplier: float | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "stickbreaking_attention"


class _Mamba2Args(BaseArgs):
    sequence_mixer_type: str = "mamba2"
    state_size: int = 128
    intermediate_size: int
    num_heads: int = 128
    conv_kernel_size: int = 4
    time_step_limit: tuple[float, float] = (0, float("inf"))
    add_bias: bool = True
    use_conv_bias: bool = True
    activation_function: str = "silu"
    num_groups: int = 8
    chunk_size: int = 256

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "mamba2"


class _GRUArgs(BaseArgs):
    sequence_mixer_type: str = "gru"
    state_size: int = 2048
    num_heads: int = 128
    add_bias: bool = True
    gradient_clipping: float | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.sequence_mixer_type == "gru"


class _RNNArgs(_GRUArgs):
    sequence_mixer_type: str = "rnn"
    activation_function: str = "tanh"
    relu_negative_slope: float | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.activation_function in ["tanh", "leaky_relu"]

        if self.activation_function == "leaky_relu":
            assert self.relu_negative_slope is not None
        else:
            assert self.relu_negative_slope is None

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
