# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import json
import os

import torch
from transformers import GenerationConfig
from transformers.generation import GenerationMixin

from .modeling_utils.softplus_decay_gate import SoftplusDecayGate
from .models import GPTBaseForCausalLM


@torch.no_grad()
def _restore_declared_dtypes(model: GPTBaseForCausalLM) -> None:
    for module in model.modules():
        if isinstance(module, SoftplusDecayGate):
            module.A_log.data = module.A_log.data.float()
            module.dt_bias.data = module.dt_bias.data.float()


_IGNORABLE_FORWARD_KWARGS = {"cache_position", "output_attentions", "output_hidden_states", "return_dict"}


class HFGPTBaseForCausalLM(GenerationMixin, GPTBaseForCausalLM):
    main_input_name = "input_ids"
    _is_stateful = True

    def __init__(self, config, **kwargs) -> HFGPTBaseForCausalLM:
        super().__init__(config, **kwargs)
        self.generation_config = GenerationConfig.from_model_config(config)

    def add_model_tags(self, tags) -> None:
        pass

    @property
    def is_gradient_checkpointing(self) -> bool:
        return any(getattr(module, "gradient_checkpointing", False) for module in self.modules())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> HFGPTBaseForCausalLM:
        if not os.path.isdir(pretrained_model_name_or_path):
            from huggingface_hub import snapshot_download

            pretrained_model_name_or_path = snapshot_download(
                pretrained_model_name_or_path, allow_patterns=["*.json", "*.safetensors", "tokenizer*"]
            )

        with open(os.path.join(pretrained_model_name_or_path, "config.json")) as f:
            config_dict = json.load(f)

        if "config" not in kwargs:
            kwargs["config"] = cls.config_class.from_dict(config_dict)

        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        _restore_declared_dtypes(model)

        return model

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_params=None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        assert past_key_values is None, "use cache_params instead of past_key_values"

        unknown_kwargs = set(kwargs) - _IGNORABLE_FORWARD_KWARGS
        assert len(unknown_kwargs) == 0, f"forward got unexpected kwargs: {unknown_kwargs}"

        return super().forward(
            input_ids=input_ids,
            cache_params=cache_params,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=0 if logits_to_keep is None else logits_to_keep,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        use_cache: bool | None = None,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        if cache_params is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cache_params": cache_params,
            "use_cache": use_cache,
            "logits_to_keep": 1,
        }

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs: dict, num_new_tokens: int = 1, **kwargs
    ) -> dict:
        model_kwargs["cache_params"] = getattr(outputs, "cache_params", None)

        if model_kwargs.get("attention_mask") is not None:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=-1
            )

        return model_kwargs
