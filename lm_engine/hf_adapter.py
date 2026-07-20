# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast as _HFCausalLMOutputWithPast

from .generation_cache import GenerationCache
from .loss import get_autoregressive_language_modeling_loss
from .mixins.dense.main import CausalLMModelMixin
from .model_config import CommonConfig
from .modeling_utils import ParameterizedEmbedding, ParameterizedLinear


_MODEL_TYPE_TO_CAUSAL_LM_CLASS: dict[str, type[CausalLMModelMixin]] = {}


class LLMAdapter_HFConfig(PretrainedConfig):
    """Wraps a lm_engine `CommonConfig` so it satisfies `transformers.PreTrainedModel`'s requirement that `config`
    be an instance of `PretrainedConfig`. Only ever used by `LLMAdapter_HF`, never for training."""

    model_type = "lm_engine_hf_adapter"

    @classmethod
    def from_common_config(cls, common_config: CommonConfig) -> LLMAdapter_HFConfig:
        kwargs = common_config.to_dict()
        kwargs.setdefault("is_decoder", True)
        kwargs.setdefault("use_cache", True)

        return cls(**kwargs)


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

        super().__init__(type(self).config_class.from_common_config(model.config))

        self.model = model

    @classmethod
    def _get_causal_lm_class(cls, config: CommonConfig) -> type[CausalLMModelMixin]:
        model_type = config.model_type
        assert model_type in _MODEL_TYPE_TO_CAUSAL_LM_CLASS, (
            f"unknown model_type ({model_type}) for LLMAdapter_HF, has `register_hf.register_model_classes` been "
            "called?"
        )
        return _MODEL_TYPE_TO_CAUSAL_LM_CLASS[model_type]

    @classmethod
    def _from_config(cls, config: CommonConfig, **kwargs) -> LLMAdapter_HF:
        model = cls._get_causal_lm_class(config)._from_config(config, **kwargs)
        return cls(model)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *model_args, config: CommonConfig | None = None, **kwargs
    ) -> LLMAdapter_HF:
        assert config is not None, "LLMAdapter_HF.from_pretrained needs `config` (an lm_engine CommonConfig)"

        model = cls._get_causal_lm_class(config).from_pretrained(
            pretrained_model_name_or_path, *model_args, config=config, **kwargs
        )
        return cls(model)

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
            use_cache=use_cache,
        )

        # loss is computed here, not by the wrapped model: the model's own loss path is tensor-parallel/aux-loss
        # aware and belongs to the training loop, whereas this adapter only ever runs plain, single-device inference
        loss = None
        if labels is not None:
            loss = get_autoregressive_language_modeling_loss(
                lm_logits=outputs.logits,
                labels=labels,
                reduction="mean",
                shift_logits_and_labels=True,
                tensor_parallel_enabled=False,
                use_padding_free_transformer=False,
            )

        return _HFCausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.cache_params,
        )


def build_hf_adapter_classes(config_class: type[CommonConfig]) -> type[LLMAdapter_HF]:
    """Builds an `LLMAdapter_HF` subclass for one lm_engine architecture, with `config_class` named to match
    `config_class` (e.g. `GPTBaseConfig`). `transformers.AutoModelForCausalLM.register` requires
    `model_class.config_class.__name__ == config_class.__name__`, and since a single `LLMAdapter_HF` wraps every
    lm_engine architecture, each one needs its own identically-named config shim to pass that check."""

    adapter_config_class = type(config_class.__name__, (LLMAdapter_HFConfig,), {})
    adapter_model_class = type(f"{config_class.__name__}LLMAdapter_HF", (LLMAdapter_HF,), {})
    adapter_model_class.config_class = adapter_config_class

    return adapter_model_class
