# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...generation_cache import GenerationCache
from ...model_config import CommonConfig
from ...modeling_utils import (
    AttentionMaskInfo,
    PositionInfo,
    get_mlp_block,
    get_normalization_function,
    get_sequence_mixer,
)


class Block(nn.Module):
    def __init__(
        self, config: CommonConfig, use_padding_free_transformer: bool, layer_idx: int, sequence_parallel: bool
    ) -> Block:
        super().__init__()

        hidden_size = config.hidden_size
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        self.sequence_mixer = get_sequence_mixer(
            config,
            True,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
            layer_idx=layer_idx,
        )

        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        self.mlp_block = get_mlp_block(
            config,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask_info: AttentionMaskInfo = AttentionMaskInfo(),
        position_info: PositionInfo = PositionInfo(),
    ) -> torch.Tensor:
        r = x

        x = self.ln_1(x)
        x = self.sequence_mixer(
            x, cache_params=cache_params, attention_mask_info=attention_mask_info, position_info=position_info
        )

        if self.m_residual is not None:
            x = x * self.m_residual

        x = x + r
        r = x

        x = self.ln_2(x)
        x = self.mlp_block(x)

        if self.m_residual is not None:
            x = x * self.m_residual

        x = x + r

        return x
