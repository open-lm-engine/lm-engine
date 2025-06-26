# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch.nn as nn

from ...config import CommonConfig
from ...modeling_utils_TP import get_mlp_block_TP, get_normalization_function_TP, get_sequence_mixer_TP
from ..dense import Block


class Block_TP(Block):
    def __init__(
        self,
        config: CommonConfig,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        self.m_residual = config.m_residual
        self.sequence_mixer_type = config.sequence_mixer_blocks[layer_idx].sequence_mixer_type

        self.ln_1 = get_normalization_function_TP(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )
        self.sequence_mixer = get_sequence_mixer_TP(
            config,
            True,
            use_padding_free_transformer=use_padding_free_transformer,
            layer_idx=layer_idx,
            sequence_parallel=sequence_parallel,
        )
        self.ln_2 = get_normalization_function_TP(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )
        self.mlp_block = get_mlp_block_TP(
            config,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
            layer_idx=layer_idx,
        )
