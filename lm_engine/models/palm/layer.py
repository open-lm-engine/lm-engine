# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...generation_cache import GenerationCache
from ...modeling_utils import (
    AttentionMaskInfo,
    PositionInfo,
    get_mlp_block,
    get_normalization_function,
    get_sequence_mixer,
)
from .config import PaLMConfig


class PaLMBlock(nn.Module):
    def __init__(
        self, config: PaLMConfig, use_padding_free_transformer: bool, layer_idx: int | None = None
    ) -> PaLMBlock:
        super().__init__()

        self.m_residual = config.m_residual

        self.ln = get_normalization_function(
            config.normalization_function, config.hidden_size, eps=config.layer_norm_epsilon
        )
        self.sequence_mixer = get_sequence_mixer(
            config,
            True,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=False,
            layer_idx=layer_idx,
        )
        self.mlp_block = get_mlp_block(
            config,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=False,
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
        x = self.ln(x)

        # NOTE we can contenate the input matrices of attention and MLP here for speedup
        # but right now we avoid it since this code is only used for accuracy benchmarking at small scale
        a = self.sequence_mixer(
            x, cache_params=cache_params, attention_mask_info=attention_mask_info, position_info=position_info
        )

        m = self.mlp_block(x)

        x = a + m
        del a, m

        if self.m_residual is not None:
            x = x * self.m_residual

        x = x + r

        return x
