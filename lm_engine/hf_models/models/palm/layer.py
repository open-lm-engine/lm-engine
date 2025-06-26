# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn

from ...cache import GenerationCache
from ...modeling_utils import get_mlp_block, get_normalization_function, get_sequence_mixer
from .config import PaLMConfig


class PaLMBlock(nn.Module):
    def __init__(self, config: PaLMConfig, use_padding_free_transformer: bool, layer_idx: int | None = None) -> None:
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
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln(hidden_states)

        # NOTE we can contenate the input matrices of attention and MLP here for speedup
        # but right now we avoid it since this code is only used for accuracy benchmarking at small scale
        attention_out = self.sequence_mixer(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        mlp_out = self.mlp_block(hidden_states)

        hidden_states = attention_out + mlp_out
        del attention_out, mlp_out

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual

        return hidden_states
