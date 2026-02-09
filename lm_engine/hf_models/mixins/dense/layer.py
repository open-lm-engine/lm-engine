# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...cache import GenerationCache
from ...config import CommonConfig
from ...modeling_utils import get_mlp_block, get_normalization_function, get_sequence_mixer


class Block(nn.Module):
    def __init__(
        self, config: CommonConfig, use_padding_free_transformer: bool, layer_idx: int | None = None
    ) -> Block:
        super().__init__()

        hidden_size = config.hidden_size
        self.m_residual = config.m_residual
        self.sequence_mixer_type = config.sequence_mixer_blocks[layer_idx].sequence_mixer_type

        self.ln_1 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )
        self.sequence_mixer = get_sequence_mixer(config, True, use_padding_free_transformer, layer_idx)
        self.ln_2 = get_normalization_function(
            config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
        )
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
        x = self.ln_1(x)

        x = self._sequence_mixer_forward(
            x=x,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
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

    def _sequence_mixer_forward(
        self,
        x: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if self.sequence_mixer_type in ["softmax_attention", "multihead_latent_attention"]:
            x = self.sequence_mixer(
                x,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        elif self.sequence_mixer_type in ["causal_convolution", "mamba2"]:
            x = self.sequence_mixer(x, cache_params=past_key_values, attention_mask=attention_mask)
        elif self.sequence_mixer_type in ["gru", "rnn"]:
            x = self.sequence_mixer(
                x,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        elif self.sequence_mixer_type == "gated_deltanet":
            # GatedDeltaNet returns (output, attentions, past_key_values)
            x = self.sequence_mixer(
                x,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            raise ValueError(f"unexpected sequence_mixer_type ({self.sequence_mixer_type})")

        return x
