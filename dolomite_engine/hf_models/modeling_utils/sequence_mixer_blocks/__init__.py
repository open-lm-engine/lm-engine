# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...config import CommonConfig
from .causal_convolution import CausalConvolution
from .flash_attention_utils import flash_attention
from .gru import GRU
from .mamba2 import Mamba2
from .multihead_latent_attention import MultiHeadLatentAttention
from .rnn import RNN
from .softmax_attention import (
    Attention,
    get_attention_head_type,
    interleave_query_key_value_tensor_for_attention,
    repeat_key_value,
    split_query_key_value_tensor_for_attention,
)
from .stickbreaking_attention import PaddingFreeSBAttention, SBAttention


def get_sequence_mixer(
    config: CommonConfig,
    causal: bool,
    use_padding_free_transformer: bool,
    layer_idx: int,
) -> Attention | Mamba2:
    block = config.sequence_mixer_blocks[layer_idx]
    sequence_mixer_type = block.sequence_mixer_type

    if sequence_mixer_type == "causal_convolution":
        return CausalConvolution(
            hidden_size=config.hidden_size,
            in_channels=block.in_channels,
            out_channels=block.out_channels,
            kernel_size=block.kernel_size,
            num_groups=block.num_groups,
            activation_function=block.activation_function,
            add_bias=block.add_bias,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            init_method=config.init_method,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
            use_padding_free_transformer=use_padding_free_transformer,
        )
    elif sequence_mixer_type == "gru":
        return GRU(
            input_size=config.hidden_size,
            state_size=block.state_size,
            output_size=config.hidden_size,
            num_heads=block.num_heads,
            add_bias=block.add_bias,
            gradient_clipping=block.gradient_clipping,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            init_method=config.init_method,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
            use_padding_free_transformer=use_padding_free_transformer,
        )
    elif sequence_mixer_type == "mamba2":
        return Mamba2(
            hidden_size=config.hidden_size,
            ssm_state_size=block.state_size,
            ssm_intermediate_size=block.intermediate_size,
            ssm_num_heads=block.num_heads,
            conv_kernel_size=block.conv_kernel_size,
            time_step_limit=block.time_step_limit,
            add_bias=block.add_bias,
            use_conv_bias=block.use_conv_bias,
            ssm_activation_function=block.activation_function,
            num_groups=block.num_groups,
            chunk_size=block.chunk_size,
            layer_norm_epsilon=config.layer_norm_epsilon,
            initializer_range=config.initializer_range,
            init_method=config.init_method,
            m_width=config.m_width,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
        )
    elif sequence_mixer_type == "multihead_latent_attention":
        return MultiHeadLatentAttention(
            hidden_size=config.hidden_size,
            query_compression_size=block.query_compression_size,
            key_value_compression_size=block.key_value_compression_size,
            num_attention_heads=block.num_attention_heads,
            head_dim=block.head_dim,
            attention_multiplier=block.attention_multiplier,
            position_embedding_type=config.position_embedding_type,
            add_bias=block.add_bias,
            softmax_dropout=block.softmax_dropout,
            dropout=block.dropout,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            causal=True,
            layer_idx=layer_idx,
            use_padding_free_transformer=use_padding_free_transformer,
            normalization_function=block.normalization_function,
            layer_norm_epsilon=config.layer_norm_epsilon,
        )
    elif sequence_mixer_type == "rnn":
        return RNN(
            input_size=config.hidden_size,
            state_size=block.state_size,
            output_size=config.hidden_size,
            num_heads=block.num_heads,
            add_bias=block.add_bias,
            gradient_clipping=block.gradient_clipping,
            activation_function=block.activation_function,
            relu_negative_slope=block.relu_negative_slope,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            init_method=config.init_method,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
            use_padding_free_transformer=use_padding_free_transformer,
        )
    else:
        sequence_mixer_kwargs = dict(
            hidden_size=config.hidden_size,
            num_attention_heads=block.num_attention_heads,
            num_key_value_heads=block.num_key_value_heads,
            attention_multiplier=block.attention_multiplier,
            position_embedding_type=config.position_embedding_type,
            add_bias=block.add_bias,
            dropout=block.dropout,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            causal=causal,
            layer_idx=layer_idx,
        )

        if sequence_mixer_type == "softmax_attention":
            return Attention(
                **sequence_mixer_kwargs,
                softmax_dropout=block.softmax_dropout,
                use_padding_free_transformer=use_padding_free_transformer,
            )
        elif sequence_mixer_type == "stickbreaking_attention":
            if use_padding_free_transformer:
                return PaddingFreeSBAttention(**sequence_mixer_kwargs)
            else:
                return SBAttention(**sequence_mixer_kwargs)
        else:
            raise ValueError(f"unexpected sequence_mixer_type ({sequence_mixer_type})")
