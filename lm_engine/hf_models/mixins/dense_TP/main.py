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
from ...modeling_utils_TP import LMHead_TP
from ..dense import CausalLMModelMixin
from ..modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    PipelineParallelInput,
    PipelineParallelOutput,
)
from .base import PreTrainedModelMixin_TP


class CausalLMModelMixin_TP(PreTrainedModelMixin_TP, CausalLMModelMixin):
    model_parallel_state_dict_function = None

    def _init_model(self, config: CommonConfig, **kwargs) -> None:
        self.vocab_size = config.vocab_size
        self.transformer = self.base_model_class(config, **kwargs)

        if self.is_last_stage:
            if not self._tied_word_embeddings:
                self.lm_head = LMHead_TP(
                    self.vocab_size,
                    config.hidden_size,
                    std=config.initializer_range,
                    use_padding_free_transformer=self.use_padding_free_transformer,
                    sequence_parallel=self.sequence_parallel,
                )

            self.m_width = config.m_width

        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | list[list[int]] | None = None,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | list[list[int]] | None = None,
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

        if self.is_pipeline_parallel_enabled:
            past_key_values = None

        clear_aux_loss()

        if self.is_first_stage:
            assert pipeline_parallel_input is None, "first stage should not get pipeline_parallel_input"
            input_ids, position_ids, token_type_ids, labels, cu_seqlens, max_seqlen = self.prepare_inputs_for_model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
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
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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
                if is_kernel_allowed(Kernel.fused_linear_cross_entropy_cute):
                    if self.m_width is not None:
                        hidden_states = hidden_states / self.m_width
                else:
                    lm_logits = self.get_lm_logits(hidden_states)

                    if self.m_width is not None:
                        lm_logits = lm_logits / self.m_width
            else:
                assert not self.is_pipeline_parallel_enabled
                assert not is_kernel_allowed(Kernel.fused_linear_cross_entropy_cute)

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
            LMHead_TP.compute_with_weight(
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
        cls, pretrained_model_name_or_path: str, torch_dtype: torch.dtype = torch.float32, **kwargs
    ) -> CausalLMModelMixin_TP:
        config: CommonConfig = cls.config_class.from_pretrained(pretrained_model_name_or_path)

        # use dummy tensors to avoid initializing model here
        with torch.device("meta"):
            # try sharding vocab matrices if really struggling for memory
            model = cls._from_config(config, **kwargs)
            model = model.to(dtype=torch_dtype)

        # copy to device without copying storage
        model = model.to_empty(device=torch.cuda.current_device())
        model.load_from_safetensors_weights_manager(SafeTensorsWeightsManager(pretrained_model_name_or_path))

        return model

    def load_from_safetensors_weights_manager(self, safetensors_weights_manager: SafeTensorsWeightsManager) -> None:
        with torch.device(torch.cuda.current_device()):
            position_embedding_type = self.config.position_embedding_type

            if position_embedding_type == "rope":
                self.transformer.rope.reset_parameters()

        state_dict = self.__class__.model_parallel_state_dict_function(
            config=self.config,
            safetensors_weights_manager=safetensors_weights_manager,
            num_pipeline_stages=self.num_pipeline_stages,
            pipeline_stage_id=self.pipeline_stage_id,
        )

        self.load_state_dict(state_dict)

    def get_dummy_input_tensor(
        self, micro_batch_size: int, sequence_length: int, intermediate_dtype: torch.dtype
    ) -> tuple[torch.Tensor] | torch.Tensor:
        if self.is_first_stage:
            # 1 is added to sequence length since megatron's dataloader gives an extra token and for good reason
            dummy_input = torch.empty(
                micro_batch_size, sequence_length + 1, device=torch.cuda.current_device(), dtype=torch.long
            )
        else:
            dummy_input = self._get_dummy_intermediate_tensor(
                micro_batch_size, sequence_length, intermediate_dtype=intermediate_dtype
            )

            dummy_input = (
                dummy_input,
                torch.empty(1, device=torch.cuda.current_device(), dtype=intermediate_dtype),
            )

        return dummy_input

    def get_dummy_output_tensor(
        self,
        micro_batch_size: int,
        sequence_length: int,
        intermediate_dtype: torch.dtype,
        output_parallel_lm_logits_if_possible: bool,
    ) -> tuple[torch.Tensor] | torch.Tensor:
        if self.is_last_stage:
            vocab_size = self.config.vocab_size
            if output_parallel_lm_logits_if_possible:
                vocab_size = divide_if_divisible(vocab_size, ProcessGroupManager.get_tensor_parallel_world_size(), "")

            if self.use_padding_free_transformer:
                tensor = torch.empty(
                    micro_batch_size * sequence_length,
                    vocab_size,
                    device=torch.cuda.current_device(),
                    dtype=intermediate_dtype,
                )
            else:
                tensor = torch.empty(
                    micro_batch_size,
                    sequence_length,
                    vocab_size,
                    device=torch.cuda.current_device(),
                    dtype=intermediate_dtype,
                )
        else:
            tensor = self._get_dummy_intermediate_tensor(
                micro_batch_size, sequence_length, intermediate_dtype=intermediate_dtype
            )

        tensor = (tensor, torch.empty(1, device=torch.cuda.current_device(), dtype=intermediate_dtype))

        return tensor

    def _get_dummy_intermediate_tensor(
        self, micro_batch_size: int, sequence_length: int, intermediate_dtype: torch.dtype
    ) -> tuple[torch.Tensor] | torch.Tensor:
        sharded_sequence_length = (
            divide_if_divisible(sequence_length, ProcessGroupManager.get_tensor_parallel_world_size(), "")
            if self.sequence_parallel
            else sequence_length
        )

        hidden_size = self.config.hidden_size

        if self.use_padding_free_transformer:
            tensor = torch.empty(
                micro_batch_size * sharded_sequence_length,
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=intermediate_dtype,
            )
        else:
            tensor = torch.empty(
                micro_batch_size,
                sharded_sequence_length,
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=intermediate_dtype,
            )

        return tensor
