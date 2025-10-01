# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import StoppingCriteriaList

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ...cache import GenerationCache
from ...config import CommonConfig
from ...loss import clear_aux_loss, get_autoregressive_language_modeling_loss, get_aux_loss, is_aux_loss_zero
from ...mask import AttentionMaskInfo
from ...modeling_utils import ParameterizedEmbedding, ParameterizedLinear
from ...utils import is_generation_cache_enabled
from ..modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .base import PreTrainedModelMixin


class CausalLMModelMixin(PreTrainedModelMixin):
    _tied_weights_keys = ["lm_head.weight"]
    base_model_class = None

    def __init__(self, config: CommonConfig, **kwargs) -> CausalLMModelMixin:
        super().__init__(config, **kwargs)

        self.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0)
        self._init_model(config, **kwargs)

    def _init_model(self, config: CommonConfig, **kwargs) -> None:
        self.transformer = self.base_model_class(config, **kwargs)

        if not self._tied_word_embeddings:
            self.lm_head = ParameterizedLinear(
                config.hidden_size, config.vocab_size, bias=False, std=config.initializer_range
            )

        self.m_width = config.m_width

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ParameterizedEmbedding:
        return self.transformer.wte

    def set_input_embeddings(self, value: ParameterizedEmbedding) -> None:
        self.transformer.wte = value

    def get_output_embeddings(self) -> ParameterizedLinear:
        return self.transformer.wte if self._tied_word_embeddings else self.lm_head

    def set_output_embeddings(self, new_embeddings: ParameterizedLinear) -> None:
        if not self._tied_word_embeddings:
            self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool = True,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        reduction: str = "mean",
    ) -> CausalLMOutputWithPast:
        assert return_dict
        assert inputs_embeds is None
        assert position_ids is not None, "max_seqlen needs to be specified when specifying cu_seqlens"
        assert attention_mask is None, "attention_mask should not be passed when specifying cu_seqlens"

        if use_cache or past_key_values is not None:
            raise NotImplementedError("KV caching is not supported with padding_free transformer")

        clear_aux_loss()

        attention_mask_info = self._get_attention_mask_info(
            x=input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, attention_mask=attention_mask
        )

        if is_generation_cache_enabled():
            past_key_values = (
                GenerationCache(self.config) if use_cache and past_key_values is None else past_key_values
            )

        transformer_outputs: BaseModelOutputWithPast = self.transformer(
            input_ids=input_ids,
            attention_mask_info=attention_mask_info,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = transformer_outputs.last_hidden_state
        past_key_values = transformer_outputs.past_key_values
        del transformer_outputs

        lm_logits = None
        loss = None

        if labels is None:
            if is_kernel_allowed(Kernel.fused_linear_cross_entropy):
                if self.m_width is not None:
                    hidden_states = hidden_states / self.m_width
            else:
                lm_logits = self.get_lm_logits(hidden_states)

                if self.m_width is not None:
                    lm_logits = lm_logits / self.m_width
        else:
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
                tensor_parallel_enabled=False,
            )

        aux_loss = get_aux_loss()

        if loss is not None and not is_aux_loss_zero(aux_loss):
            loss = loss + self.router_aux_loss_coef * aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=lm_logits,
            past_key_values=past_key_values,
            last_hidden_state=hidden_states,
        )

    def get_lm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return (
            F.linear(hidden_states, self.transformer.wte.weight)
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

    def _get_attention_mask_info(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> AttentionMaskInfo:
        kwargs = {}
        if cu_seqlens is None:
            if attention_mask is None:
                kwargs["batch_size"] = x.size(0)
                kwargs["max_seqlen"] = x.size(1)
                kwargs["device"] = x.device
            else:
                kwargs["attention_mask"] = attention_mask
        else:
            kwargs["cu_seqlens"] = cu_seqlens
            kwargs["max_seqlen"] = max_seqlen

        return AttentionMaskInfo(**kwargs)
