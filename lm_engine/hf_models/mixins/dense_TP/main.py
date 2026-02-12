# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed._tensor.placement_types import Replicate, Shard

from ....dtensors import dtensor_to_tensor, tensor_to_dtensor
from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import ProcessGroupManager, SafeTensorsWeightsManager, divide_if_divisible
from ...cache import GenerationCache
from ...config import CommonConfig
from ...loss import (
    add_aux_loss,
    clear_aux_loss,
    get_autoregressive_language_modeling_loss,
    get_aux_loss,
    is_aux_loss_zero,
)
from ...modeling_utils import LMHead
from ...parameter import _INIT_MARKER, get_parameter_marker_maps, set_parameter_marker_maps
from ..dense import CausalLMModelMixin
from ..modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    PipelineParallelInput,
    PipelineParallelOutput,
)


class CausalLMModelMixin_TP(CausalLMModelMixin):
    model_parallel_state_dict_function = None

    def forward(
        self,
        input_ids: torch.Tensor | list[list[int]] | None = None,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | list[list[int]] | None = None,
        inputs_embeds: torch.Tensor | list[list[float]] | None = None,
        labels: torch.Tensor | list[list[int]] | None = None,
        use_cache: bool | None = None,
        return_dict: bool = True,
        output_parallel_lm_logits: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        reduction: str = "mean",
        pipeline_parallel_input: PipelineParallelInput | None = None,
    ) -> CausalLMOutputWithPast | PipelineParallelOutput:
        assert return_dict
        assert inputs_embeds is None

        if self.is_pipeline_parallel_enabled:
            past_key_values = None

        clear_aux_loss()

        if self.is_first_stage:
            assert pipeline_parallel_input is None, "first stage should not get pipeline_parallel_input"
            input_ids, position_ids, labels, cu_seqlens, max_seqlen = self.prepare_inputs_for_model(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
        else:
            assert input_ids is None
            add_aux_loss(pipeline_parallel_input.aux_loss)

        transformer_outputs: BaseModelOutputWithPast = self.transformer(
            input_ids=input_ids if pipeline_parallel_input is None else pipeline_parallel_input.hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        hidden_states = transformer_outputs.last_hidden_state
        past_key_values = transformer_outputs.past_key_values

        del pipeline_parallel_input
        del transformer_outputs

        lm_logits = None
        loss = None
        aux_loss = get_aux_loss()

        if self.is_last_stage:
            if labels is None:
                if is_kernel_allowed(Kernel.fused_linear_cross_entropy):
                    if self.m_width is not None:
                        hidden_states = hidden_states / self.m_width
                else:
                    lm_logits = self.get_lm_logits(hidden_states)

                    if self.m_width is not None:
                        lm_logits = lm_logits / self.m_width
            else:
                assert not self.is_pipeline_parallel_enabled
                assert not is_kernel_allowed(Kernel.fused_linear_cross_entropy)

                lm_logits = self.get_lm_logits(hidden_states)

                if self.m_width is not None:
                    lm_logits = lm_logits / self.m_width

                loss = get_autoregressive_language_modeling_loss(
                    lm_logits=lm_logits,
                    labels=labels,
                    hidden_states=None,
                    vocab_weight=None,
                    cu_seqlens=cu_seqlens,
                    use_padding_free_transformer=self.use_padding_free_transformer,
                    reduction=reduction,
                    shift_logits_and_labels=True,
                    tensor_parallel_enabled=ProcessGroupManager.is_tensor_parallel_enabled(),
                )

            if not output_parallel_lm_logits:
                # all gather
                lm_logits = tensor_to_dtensor(lm_logits, device_mesh=self.tp_mesh, current_placement=Shard(-1))
                lm_logits = dtensor_to_tensor(lm_logits, device_mesh=self.tp_mesh, desired_placement=Replicate())

            if loss is not None and not is_aux_loss_zero(aux_loss):
                loss = loss + self.router_aux_loss_coef * aux_loss

            output = CausalLMOutputWithPast(
                loss=loss,
                aux_loss=aux_loss,
                logits=lm_logits,
                past_key_values=past_key_values,
                last_hidden_state=hidden_states,
            )
        else:
            output = PipelineParallelOutput(hidden_states=hidden_states, aux_loss=aux_loss)

        return output

    def get_lm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return (
            LMHead.compute_with_weight(
                hidden_states,
                weight=self.transformer.wte.weight,
                use_padding_free_transformer=self.use_padding_free_transformer,
                sequence_parallel=self.sequence_parallel,
                tp_mesh=self.tp_mesh,
            )
            if self._tied_word_embeddings
            else self.lm_head(hidden_states)
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, dtype: torch.dtype = torch.float32, **kwargs
    ) -> CausalLMModelMixin_TP:
        config: CommonConfig = cls.config_class.from_pretrained(pretrained_model_name_or_path)

        # use dummy tensors to avoid initializing model here
        with torch.device("meta"):
            # try sharding vocab matrices if really struggling for memory
            model = cls._from_config(config, **kwargs)
            marker_maps = get_parameter_marker_maps([model], extra_markers=[_INIT_MARKER])

            model = model.to(dtype=dtype)

        # copy to device without copying storage
        model = model.to_empty(device=torch.cuda.current_device())
        set_parameter_marker_maps([model], marker_maps)

        model.load_from_safetensors_weights_manager(SafeTensorsWeightsManager(pretrained_model_name_or_path))

        return model
