# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...cache import GenerationCache
from ...modeling_utils import get_mlp_block, get_normalization_function, get_sequence_mixer
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
        self.sequence_mixer = get_sequence_mixer(config, True, use_padding_free_transformer, layer_idx)
        self.mlp_block = get_mlp_block(
            config, use_padding_free_transformer=use_padding_free_transformer, layer_idx=layer_idx
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        r = x
        x = self.ln(x)

        # NOTE we can contenate the input matrices of attention and MLP here for speedup
        # but right now we avoid it since this code is only used for accuracy benchmarking at small scale
        a = self.sequence_mixer(
            x,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        m = self.mlp_block(x)

        x = a + m
        del a, m

        if self.m_residual is not None:
            x = x * self.m_residual

        x = x + r

        return x
