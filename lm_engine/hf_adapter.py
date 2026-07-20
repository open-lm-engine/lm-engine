# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast as _HFCausalLMOutputWithPast

from .generation_cache import GenerationCache
from .mixins.dense.main import CausalLMModelMixin
from .model_config import CommonConfig
from .modeling_utils import ParameterizedEmbedding, ParameterizedLinear


class LLMAdapter_HFConfig(PretrainedConfig):
    """Wraps a lm_engine `CommonConfig` so it satisfies `transformers.PreTrainedModel`'s requirement that `config`
    be an instance of `PretrainedConfig`. Only ever used by `LLMAdapter_HF`, never for training."""

    model_type = "lm_engine_hf_adapter"

    def __init__(self, common_config: CommonConfig | None = None, **kwargs) -> LLMAdapter_HFConfig:
        if common_config is not None:
            kwargs = {**common_config.to_dict(), **kwargs}

        kwargs.setdefault("is_decoder", True)
        kwargs.setdefault("use_cache", True)

        super().__init__(**kwargs)


class LLMAdapter_HF(PreTrainedModel, GenerationMixin):
    """Adapts an already-built lm_engine `CausalLMModelMixin` model (e.g. `GPTBaseForCausalLM`) to the
    `transformers.PreTrainedModel` + `GenerationMixin` interface, so it can be dropped into `model.generate(...)`,
    `transformers.pipeline(...)`, or any HuggingFace-based evaluation harness without modification.

    This is an inference-only adapter. It does not support training (no pipeline/tensor/sequence parallelism,
    no padding-free packed sequences, no loss-scaled aux loss handling): use the wrapped lm_engine model, its own
    `mixins.dense.main.CausalLMModelMixin.generate`, and the lm_engine training loop directly for anything else.

    Example:
        model = GPTBaseForCausalLM.from_pretrained(path)
        hf_model = LLMAdapter_HF(model)
        hf_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=20)
    """

    config_class = LLMAdapter_HFConfig
    base_model_prefix = "model"

    def __init__(self, model: CausalLMModelMixin) -> LLMAdapter_HF:
        assert not model.is_pipeline_parallel_enabled, "LLMAdapter_HF does not support pipeline parallelism"
        assert not model.use_padding_free_transformer, "LLMAdapter_HF does not support the padding-free transformer"

        super().__init__(LLMAdapter_HFConfig(model.config))

        self.model = model

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # lm_engine models manage their own `GenerationCache`, created lazily on the first forward call when
        # `cache_params` is None. Returning False here stops `generate()` from pre-instantiating a `DynamicCache`
        # for us, the same escape hatch HF uses for architectures with a non-standard cache (e.g. Mamba).
        return False

    def can_generate(self) -> bool:
        return True

    def get_input_embeddings(self) -> ParameterizedEmbedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: ParameterizedEmbedding) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> ParameterizedLinear:
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: ParameterizedLinear) -> None:
        self.model.set_output_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: GenerationCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,  # swallows extra kwargs GenerationMixin forwards (cache_position, output_attentions, ...)
    ) -> _HFCausalLMOutputWithPast:
        assert inputs_embeds is None, "LLMAdapter_HF does not support inputs_embeds"

        outputs = self.model(
            input_ids=input_ids,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=use_cache,
        )

        return _HFCausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.cache_params,
        )
