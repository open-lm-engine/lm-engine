# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ...cache import GenerationCache
from ...config import CommonConfig
from ...loss import clear_aux_loss, get_autoregressive_language_modeling_loss, get_aux_loss, is_aux_loss_zero
from ...modeling_utils import ParameterizedEmbedding, ParameterizedLinear
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
        input_ids: torch.Tensor | list[list[int]] | None = None,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | list[list[int]] | None = None,
        inputs_embeds: torch.Tensor | list[list[float]] | None = None,
        labels: torch.Tensor | list[list[int]] | None = None,
        use_cache: bool | None = None,
        return_dict: bool = True,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        reduction: str = "mean",
    ) -> CausalLMOutputWithPast:
        assert return_dict
        assert inputs_embeds is None

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

        clear_aux_loss()

        transformer_outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        hidden_states = transformer_outputs.last_hidden_state
        past_key_values = transformer_outputs.past_key_values
        del transformer_outputs

        lm_logits = None
        loss = None

        if labels is None:
            if is_kernel_allowed(Kernel.fused_linear_cross_entropy_cute):
                if self.m_width is not None:
                    hidden_states = hidden_states / self.m_width
            else:
                lm_logits = self.get_lm_logits(hidden_states)

                if self.m_width is not None:
                    lm_logits = lm_logits / self.m_width
        else:
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
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        assert not self.use_padding_free_transformer

        has_attention_mask = attention_mask is not None
        is_greedy = temperature is None or temperature == 0

        # prefill
        output = self(input_ids=input_ids, attention_mask=attention_mask)

        # decode
        generated_tokens = [input_ids]
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

            lm_logits: torch.Tensor = output.logits[:, -1, :]

            if is_greedy:
                next_token = lm_logits.argmax(dim=-1).unsqueeze(1)
            else:
                if temperature is not None:
                    lm_logits = lm_logits / temperature

                if top_k is not None and top_k < lm_logits.size(-1):
                    # mask all tokens with logits less than the min(topk(lm_logits))
                    lm_logits_top_k_min = lm_logits.topk(k=top_k)[0][:, -1]

                probabilities = F.softmax(lm_logits, dim=-1)

            generated_tokens.append(next_token)

            output = self(input_ids=next_token, attention_mask=attention_mask, past_key_values=output.past_key_values)

        generated_tokens = torch.cat(generated_tokens, dim=-1)

        return generated_tokens
