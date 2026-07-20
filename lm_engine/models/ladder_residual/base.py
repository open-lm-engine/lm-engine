# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...generation_cache import GenerationCache
from ...mixins import BaseModelMixin, PreTrainedModelMixin
from ...modeling_utils import AttentionMaskInfo, BaseModelOutputWithPast, PositionInfo
from ...utils import is_generation_cache_enabled
from .config import LadderResidualConfig
from .layer import LadderResidualBlock


class LadderResidualPreTrainedModel(PreTrainedModelMixin):
    config_class = LadderResidualConfig
    layer_class = LadderResidualBlock
    _no_split_modules = ["LadderResidualBlock"]


class LadderResidualModel(LadderResidualPreTrainedModel, BaseModelMixin):
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        cache_params: GenerationCache | None = None,
        attention_mask_info: AttentionMaskInfo = AttentionMaskInfo(),
        position_info: PositionInfo = PositionInfo(),
    ) -> BaseModelOutputWithPast:
        hidden_states, attention_mask_info, position_info, cache_params = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            cache_params=cache_params,
            attention_mask_info=attention_mask_info,
            position_info=position_info,
        )

        current_attention_out = None
        current_mlp_out = None

        if is_generation_cache_enabled() and cache_params is None:
            cache_params = GenerationCache()

        for block in self.h:
            current_attention_out, current_mlp_out, hidden_states = block(
                current_attention_out=current_attention_out,
                current_mlp_out=current_mlp_out,
                residual=hidden_states,
                cache_params=cache_params,
                attention_mask_info=attention_mask_info,
                position_info=position_info,
            )

        hidden_states = hidden_states + current_attention_out + current_mlp_out
        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, cache_params=cache_params)
