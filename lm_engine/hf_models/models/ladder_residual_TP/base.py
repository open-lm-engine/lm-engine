# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.distributed._tensor.placement_types import Partial, Replicate

from ....dtensors import dtensor_to_tensor, tensor_to_dtensor
from ....utils import ProcessGroupManager
from ...cache import GenerationCache
from ...mixins import BaseModelMixin_TP, BaseModelOutputWithPast, PreTrainedModelMixin_TP
from ...utils import is_generation_cache_enabled
from ..ladder_residual import LadderResidualConfig
from .layer import LadderResidualBlock_TP


class LadderResidualPreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = LadderResidualConfig
    layer_class = LadderResidualBlock_TP
    _no_split_modules = ["LadderResidualBlock_TP"]

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        if self.num_pipeline_stages > 1:
            raise NotImplementedError("pipeline parallel is not supported with this model architecture")


class LadderResidualModel_TP(LadderResidualPreTrainedModel_TP, BaseModelMixin_TP):
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

        current_attention_out = None
        current_mlp_out = None

        if is_generation_cache_enabled():
            past_key_values = (
                GenerationCache(self.config) if use_cache and past_key_values is None else past_key_values
            )

        for layer_idx in range(self.layer_start_id, self.layer_end_id):
            current_attention_out, current_mlp_out, hidden_states = self.h[str(layer_idx)](
                current_attention_out=current_attention_out,
                current_mlp_out=current_mlp_out,
                residual=hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        # all reduce using DTensor API
        current_mlp_out = dtensor_to_tensor(
            tensor_to_dtensor(
                current_mlp_out,
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                current_placement=Partial(),
                desired_placement=Replicate(),
            )
        )

        hidden_states = hidden_states + current_attention_out + current_mlp_out
        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)
