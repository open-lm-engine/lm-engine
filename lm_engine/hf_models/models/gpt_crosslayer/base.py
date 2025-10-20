# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from ...cache import GenerationCache
from ...mask import AttentionMaskInfo
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
        input_ids: torch.Tensor,
        attention_mask_info: AttentionMaskInfo,
        past_key_values: GenerationCache | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
        hidden_states = self._get_initial_hidden_state(input_ids, position_ids)
        rope_cos_sin = self._get_rope_cos_sin(
            attention_mask_info.get_max_seqlen(), position_ids, dtype=hidden_states.dtype
        )

        key = None
        value = None

        for block in self.h:
            hidden_states, key, value = block(
                hidden_states=hidden_states,
                key=key,
                value=value,
                attention_mask_info=attention_mask_info,
                past_key_values=past_key_values,
                rope_cos_sin=rope_cos_sin,
            )

        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)
