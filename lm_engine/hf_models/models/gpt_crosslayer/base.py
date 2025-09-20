# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from ...cache import GenerationCache
from ...mixins import BaseModelMixin, BaseModelOutputWithPast, PreTrainedModelMixin
from .config import GPTCrossLayerConfig
from .layer import GPTCrossLayerBlock


class GPTCrossLayerPreTrainedModel(PreTrainedModelMixin):
    config_class = GPTCrossLayerConfig
    layer_class = GPTCrossLayerBlock
    _no_split_modules = ["GPTCrossLayerBlock"]

    def __init__(self, config: GPTCrossLayerConfig, *args, **kwargs) -> GPTCrossLayerPreTrainedModel:
        self.sharing_pattern = config.sharing_pattern
        super().__init__(config, *args, **kwargs)


class GPTCrossLayerModel(GPTCrossLayerPreTrainedModel, BaseModelMixin):
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        cache_params: GenerationCache | None = None,
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
            cache_params,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            cache_params=cache_params,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        cache_params = GenerationCache(self.config) if use_cache and cache_params is None else cache_params

        key = None
        value = None

        for block in self.h:
            hidden_states, key, value = block(
                hidden_states,
                key=key,
                value=value,
                cache_params=cache_params,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        del key, value
        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, cache_params=cache_params)
