# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...cache import GenerationCache
from ...mixins import BaseModelMixin, BaseModelOutputWithPast, PreTrainedModelMixin
from .config import GPTCrossLayerConfig
from .layer import GPTCrossLayerBlock


class GPTCrossLayerPreTrainedModel(PreTrainedModelMixin):
    config_class = GPTCrossLayerConfig
    layer_class = GPTCrossLayerBlock
    _no_split_modules = ["GPTCrossLayerBlock"]

    def __init__(self, config: GPTCrossLayerConfig, *args, **kwargs):
        self.sharing_pattern = config.sharing_pattern
        super().__init__(config, *args, **kwargs)


class GPTCrossLayerModel(GPTCrossLayerPreTrainedModel, BaseModelMixin):
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
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
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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
