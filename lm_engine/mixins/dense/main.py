# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from ...dtensors import dtensor_to_tensor, tensor_to_dtensor
from ...enums import Kernel
from ...generation_cache import GenerationCache
from ...kernels import is_kernel_allowed
from ...loss import add_aux_loss, clear_aux_loss, get_aux_loss
from ...model_config import CommonConfig
from ...modeling_utils import AttentionMaskInfo, DTensorModule, LMHead, PositionInfo
from ...modeling_utils.io import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    PipelineParallelInput,
    PipelineParallelOutput,
)
from ...parallel import ProcessGroupManager
from ...utils import SafeTensorsWeightsManager, divide_if_divisible
from .base import PreTrainedModelMixin


class CausalLMModelMixin(PreTrainedModelMixin, DTensorModule):
    base_model_class = None

    def __init__(self, config: CommonConfig, **kwargs) -> CausalLMModelMixin:
        super().__init__(config, **kwargs)

        self.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0)
        self._init_model(config, **kwargs)

    def _init_model(self, config: CommonConfig, **kwargs) -> None:
        self.vocab_size = config.vocab_size
        self.transformer = self.base_model_class(config, **kwargs)

        if self.is_last_stage:
            if not self._tied_word_embeddings:
                self.lm_head = LMHead(
                    self.vocab_size,
                    config.hidden_size,
                    std=config.initializer_range,
                    use_padding_free_transformer=self.use_padding_free_transformer,
                    sequence_parallel=self.sequence_parallel,
                )

            self.m_width = config.m_width

    def forward(
        self,
        input_ids: torch.Tensor | list[list[int]] | None = None,
        cache_params: GenerationCache | None = None,
        attention_mask_info: AttentionMaskInfo = AttentionMaskInfo(),
        position_info: PositionInfo = PositionInfo(),
        output_parallel_lm_logits: bool = False,
        pipeline_parallel_input: PipelineParallelInput | None = None,
    ) -> CausalLMOutputWithPast | PipelineParallelOutput:
        if self.is_pipeline_parallel_enabled:
            assert cache_params is None

        clear_aux_loss()

        if self.is_first_stage:
            assert pipeline_parallel_input is None, "first stage should not get pipeline_parallel_input"

            if self.use_padding_free_transformer:
                assert (
                    attention_mask_info.cu_seqlens is not None
                ), "cu_seqlens needs to be specified when using tensor inputs with padding_free transformer"
                assert (
                    position_info.position_ids is not None
                ), "position_ids needs to be specified when specifying cu_seqlens"
                assert (
                    attention_mask_info.max_seqlen is not None
                ), "max_seqlen needs to be specified when specifying cu_seqlens"
                assert (
                    attention_mask_info.attention_mask is None
                ), "attention_mask should not be passed when specifying cu_seqlens"

                if cache_params is not None:
                    raise NotImplementedError("KV caching is not supported with padding_free transformer")
        else:
            assert input_ids is None
            add_aux_loss(pipeline_parallel_input.aux_loss)

        transformer_outputs: BaseModelOutputWithPast = self.transformer(
            input_ids=input_ids if pipeline_parallel_input is None else pipeline_parallel_input.hidden_states,
            cache_params=cache_params,
            attention_mask_info=attention_mask_info,
            position_info=position_info,
        )

        hidden_states = transformer_outputs.last_hidden_state
        cache_params = transformer_outputs.cache_params

        del pipeline_parallel_input
        del transformer_outputs

        lm_logits = None
        aux_loss = get_aux_loss()

        if self.is_last_stage:
            if is_kernel_allowed(Kernel.fused_linear_cross_entropy):
                if self.m_width is not None:
                    hidden_states = hidden_states * (1 / self.m_width)
            else:
                lm_logits = (
                    LMHead.compute_with_weight(
                        x=hidden_states,
                        weight=self.transformer.wte.weight,
                        use_padding_free_transformer=self.use_padding_free_transformer,
                        sequence_parallel=self.sequence_parallel,
                        tp_mesh=self.tp_mesh if self.is_tp_enabled else None,
                    )
                    if self._tied_word_embeddings
                    else self.lm_head(hidden_states)
                )

                if self.m_width is not None:
                    lm_logits = lm_logits * (1 / self.m_width)

            if self.is_tp_enabled and not output_parallel_lm_logits:
                # all gather
                lm_logits = tensor_to_dtensor(lm_logits, device_mesh=self.tp_mesh, current_placement=Shard(-1))
                lm_logits = dtensor_to_tensor(lm_logits, device_mesh=self.tp_mesh, desired_placement=Replicate())

            output = CausalLMOutputWithPast(
                aux_loss=aux_loss,
                logits=lm_logits,
                cache_params=cache_params,
                last_hidden_state=hidden_states,
            )
        else:
            output = PipelineParallelOutput(hidden_states=hidden_states, aux_loss=aux_loss)

        return output

    def load_from_safetensors_weights_manager(self, safetensors_weights_manager: SafeTensorsWeightsManager) -> None:
        with torch.device(torch.cuda.current_device()):
            position_embedding_type = self.config.position_embedding_type

            if position_embedding_type == "rope":
                self.transformer.rope.reset_parameters()

        state_dict = {}
        for name, parameter in list(self.named_parameters()) + list(self.named_buffers()):
            if not safetensors_weights_manager.has_tensor(name):
                continue

            p = safetensors_weights_manager.get_tensor(tensor_name=name, dtype=parameter.dtype)
            if isinstance(parameter, DTensor):
                p = distribute_tensor(tensor=p, device_mesh=parameter.device_mesh, placements=parameter.placements)

            state_dict[name] = p

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
            vocab_size = self.vocab_size
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
