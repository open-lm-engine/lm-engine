# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate, Shard
from transformers import StoppingCriteriaList

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
from ...modeling_utils import DTensorModule, LMHead
from ...parameter import _INIT_MARKER, get_parameter_marker_maps, set_parameter_marker_maps
from ..modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    PipelineParallelInput,
    PipelineParallelOutput,
)
from .base import PreTrainedModelMixin


class CausalLMModelMixin(PreTrainedModelMixin, DTensorModule):
    base_model_class = None
    model_parallel_state_dict_function = None

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
            assert past_key_values is None

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
                    tensor_parallel_enabled=self.is_tp_enabled,
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

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 20,
        temperature: float = 0,
        top_k: int | None = None,
        top_p: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        assert not self.use_padding_free_transformer

        has_attention_mask = attention_mask is not None
        min_tokens_to_keep = 1

        # for HF compatibility
        if "max_length" in kwargs:
            max_new_tokens = kwargs.pop("max_length") - (
                input_ids.size(-1) if attention_mask is None else attention_mask.sum(dim=-1).min().item()
            )

        pad_token_id = kwargs.pop("pad_token_id", self.generation_config.pad_token_id)
        if pad_token_id is None:
            pad_token_id = self.generation_config.eos_token_id

        kwargs.pop("use_cache", None)

        if "do_sample" in kwargs:
            if kwargs.pop("do_sample"):
                if temperature == 0:
                    temperature = 1
            else:
                temperature = 0

        stopping_criteria_list = kwargs.pop("stopping_criteria", None)
        if stopping_criteria_list is not None:
            stopping_criteria_list = StoppingCriteriaList(stopping_criteria_list)

        assert len(kwargs) == 0

        # prefill
        output = self(input_ids=input_ids, attention_mask=attention_mask)
        finished = torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.bool)

        # decode
        generated_tokens = input_ids
        for num_generated_tokens in range(max_new_tokens):
            if has_attention_mask:
                attention_mask = torch.cat(
                    (attention_mask, torch.ones(input_ids.size(0), 1, device=input_ids.device, dtype=torch.int32)),
                    dim=-1,
                )
            else:
                attention_mask = torch.ones(
                    input_ids.size(0),
                    input_ids.size(1) + num_generated_tokens + 1,
                    device=input_ids.device,
                    dtype=torch.int32,
                )

            lm_logits = output.logits[:, -1, :]
            past_key_values = output.past_key_values

            if temperature == 0:
                next_token = lm_logits.argmax(dim=-1).unsqueeze(1)
            else:
                if temperature != 1:
                    lm_logits = lm_logits / temperature

                if top_k is not None and top_k < lm_logits.size(-1):
                    # mask all tokens with logits less than the min(topk(lm_logits))
                    lm_logits_top_k_min = lm_logits.topk(k=top_k)[0][:, -1].unsqueeze(-1)
                    mask = lm_logits < lm_logits_top_k_min
                    lm_logits = lm_logits.masked_fill(mask, -float("inf"))

                if top_p is not None:
                    sorted_logits, sorted_indices = lm_logits.sort(descending=False)
                    cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

                    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
                    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                    # Keep at least min_tokens_to_keep
                    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

                    # scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    lm_logits = lm_logits.masked_fill(indices_to_remove, -float("inf"))

                probabilities = F.softmax(lm_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            next_token = next_token.masked_fill(finished.unsqueeze(1), pad_token_id)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            finished = finished | (next_token.squeeze(1) == self.generation_config.eos_token_id)
            if stopping_criteria_list is not None:
                finished = finished | stopping_criteria_list(generated_tokens, None)

            # early exit when all sequences finish
            if finished.min() == 1:
                break

            output: CausalLMOutputWithPast = self(
                input_ids=next_token, attention_mask=attention_mask, past_key_values=past_key_values
            )

        return generated_tokens

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, dtype: torch.dtype = torch.float32, **kwargs
    ) -> CausalLMModelMixin:
        if ProcessGroupManager.is_tensor_parallel_enabled():
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
        else:
            model = super().from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path, dtype=dtype, **kwargs
            )

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
