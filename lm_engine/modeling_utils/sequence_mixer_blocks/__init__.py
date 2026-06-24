# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import TYPE_CHECKING

from ...parallel import ProcessGroupManager
from .gated_deltanet import GatedDeltaNet, GatedDeltaNetArgs
from .gru import GRU, GRUArgs
from .m2rnn import M2RNN, M2RNNArgs
from .mamba2 import Mamba2, Mamba2Args
from .rnn import RNN, RNNArgs
from .softmax_attention import (
    SoftmaxAttention,
    SoftmaxAttentionArgs,
    flash_attention,
    interleave_query_key_value_tensor_for_attention,
    split_query_key_value_tensor_for_attention,
)


if TYPE_CHECKING:
    from ...model_config import CommonConfig

SEQUENCE_MIXER_TYPE = SoftmaxAttention | GRU | Mamba2 | RNN | GatedDeltaNet


def get_sequence_mixer(
    config: CommonConfig, causal: bool, use_padding_free_transformer: bool, sequence_parallel: bool, layer_idx: int
) -> SEQUENCE_MIXER_TYPE:
    block = config.sequence_mixer_blocks[layer_idx]
    sequence_mixer_type = block.sequence_mixer_type

    is_tp_enabled = ProcessGroupManager.is_tensor_parallel_enabled()

    if sequence_mixer_type == "gru":
        assert not is_tp_enabled
        return GRU(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            config=block,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            init_method=config.init_method,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
            use_depth_scaled_init=config.use_depth_scaled_init,
            use_padding_free_transformer=use_padding_free_transformer,
        )
    elif sequence_mixer_type == "rnn":
        assert not is_tp_enabled
        return RNN(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            config=block,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            init_method=config.init_method,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
            use_depth_scaled_init=config.use_depth_scaled_init,
            use_padding_free_transformer=use_padding_free_transformer,
        )
    elif sequence_mixer_type == "m2rnn":
        return M2RNN(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            config=block,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            init_method=config.init_method,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
            use_depth_scaled_init=config.use_depth_scaled_init,
            use_padding_free_transformer=use_padding_free_transformer,
        )
    elif sequence_mixer_type == "mamba2":
        assert not is_tp_enabled
        return Mamba2(
            hidden_size=config.hidden_size,
            config=block,
            layer_norm_epsilon=config.layer_norm_epsilon,
            initializer_range=config.initializer_range,
            init_method=config.init_method,
            m_width=config.m_width,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
            use_depth_scaled_init=config.use_depth_scaled_init,
        )
    elif sequence_mixer_type == "gated_deltanet":
        assert not is_tp_enabled
        return GatedDeltaNet(
            hidden_size=config.hidden_size,
            config=block,
            layer_idx=layer_idx,
            norm_eps=config.layer_norm_epsilon,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            use_depth_scaled_init=config.use_depth_scaled_init,
            use_padding_free_transformer=use_padding_free_transformer,
        )
    elif sequence_mixer_type == "softmax_attention":
        return SoftmaxAttention(
            hidden_size=config.hidden_size,
            config=block,
            position_embedding_type=config.position_embedding_type,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            causal=causal,
            layer_idx=layer_idx,
            use_depth_scaled_init=config.use_depth_scaled_init,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )
    else:
        raise ValueError(f"unexpected sequence_mixer_type ({sequence_mixer_type})")
