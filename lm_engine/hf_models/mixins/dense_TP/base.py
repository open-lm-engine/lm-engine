# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from ....utils import ProcessGroupManager
from ...cache import GenerationCache
from ...utils import is_generation_cache_enabled
from ..dense import BaseModelMixin
from ..modeling_outputs import BaseModelOutputWithPast


class BaseModelMixin_TP(BaseModelMixin):
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
        if self.is_first_stage:
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
        else:
            assert past_key_values is None
            assert attention_mask is None

            hidden_states = input_ids
            past_length = 0

            if self.use_padding_free_transformer:
                key_length = max_seqlen
                # query length will change if past_key_values is not None
                query_length = key_length - past_length
            else:
                key_length = (
                    hidden_states.size(1) * ProcessGroupManager.get_tensor_parallel_world_size()
                    if self.sequence_parallel
                    else hidden_states.size(1)
                )
                query_length = key_length - past_length

            position_ids = torch.arange(past_length, key_length, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).view(-1, query_length)

            rope_cos_sin = self._get_rope_cos_sin(key_length, position_ids, dtype=hidden_states.dtype)

        if is_generation_cache_enabled():
            past_key_values = (
                GenerationCache(self.config) if use_cache and past_key_values is None else past_key_values
            )

        for layer_idx in range(self.layer_start_id, self.layer_end_id):
            hidden_states = self.h[str(layer_idx)](
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        if self.is_last_stage:
            hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)
