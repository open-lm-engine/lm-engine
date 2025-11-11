# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...cache import GenerationCache
from ...mixins import BaseModelMixin, BaseModelOutputWithPast, PreTrainedModelMixin
from ...modeling_utils import ParameterizedEmbedding, get_normalization_function
from .config import GPTCrossLayerConfig
from .layer import GPTCrossLayerBlock


class GPTCrossLayerPreTrainedModel(PreTrainedModelMixin):
    config_class = GPTCrossLayerConfig
    layer_class = GPTCrossLayerBlock
    _no_split_modules = ["GPTCrossLayerBlock"]

    def __init__(self, config: GPTCrossLayerConfig, *args, **kwargs) -> GPTCrossLayerPreTrainedModel:
        self.sharing_pattern = config.sharing_pattern
        super().__init__(config, *args, **kwargs)
        assert self.use_padding_free_transformer


class GPTCrossLayerModel(GPTCrossLayerPreTrainedModel, BaseModelMixin):
    def _init_model(self, config: GPTCrossLayerConfig, **kwargs) -> None:
        self.embed_dim = config.hidden_size
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range
        self.sequence_mixer_block_types = [
            config.sequence_mixer_blocks[i].sequence_mixer_type for i in range(config.num_layers)
        ]

        self.wte = ParameterizedEmbedding(config.vocab_size, self.embed_dim, std=self.initializer_range)

        self.embedding_dropout = (
            nn.Identity() if config.embedding_dropout == 0 else nn.Dropout(config.embedding_dropout)
        )
        self.h = nn.ModuleList([self.layer_class(config, layer_idx=i) for i in range(config.num_layers)])
        self.ln_f = get_normalization_function(
            config.normalization_function, self.embed_dim, eps=config.layer_norm_epsilon
        )

        self.rope_dim = config.rope_dim

        self.position_embedding_type = config.position_embedding_type
        self._setup_positional_encoding()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> BaseModelOutputWithPast:
        (
            use_cache,
            hidden_states,
            attention_mask,
            position_ids,
            rope_cos_sin,
            past_key_values,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        past_key_values = GenerationCache(self.config) if use_cache and past_key_values is None else past_key_values

        key = None
        value = None

        for block in self.h:
            hidden_states, key, value = block(
                hidden_states,
                key=key,
                value=value,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        del key, value
        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)
