# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .hf_adapter import _MODEL_TYPE_TO_CAUSAL_LM_CLASS, build_hf_adapter_classes
from .mixins.dense.main import CausalLMModelMixin
from .models import (
    GPTBaseConfig,
    GPTBaseForCausalLM,
    GPTBaseModel,
    GPTCrossLayerConfig,
    GPTCrossLayerForCausalLM,
    GPTCrossLayerModel,
    LadderResidualConfig,
    LadderResidualForCausalLM,
    LadderResidualModel,
    PaLMConfig,
    PaLMForCausalLM,
    PaLMModel,
)


# (AutoConfig, AutoModel, AutoModelForCausalLM)
_CUSTOM_MODEL_REGISTRY = [
    (GPTBaseConfig, GPTBaseModel, GPTBaseForCausalLM),
    (GPTCrossLayerConfig, GPTCrossLayerModel, GPTCrossLayerForCausalLM),
    (LadderResidualConfig, LadderResidualModel, LadderResidualForCausalLM),
    (PaLMConfig, PaLMModel, PaLMForCausalLM),
]
_CUSTOM_MODEL_TYPES = []
_CUSTOM_MODEL_CLASSES = []


def register_model_classes() -> None:
    for config_class, auto_model_class, auto_model_for_causal_lm_class in _CUSTOM_MODEL_REGISTRY:
        model_type = config_class.model_fields["model_type"].default

        AutoConfig.register(model_type, config_class)
        AutoModel.register(config_class, auto_model_class)
        AutoModelForCausalLM.register(config_class, build_hf_adapter_classes(config_class))

        _CUSTOM_MODEL_TYPES.append(model_type)
        _CUSTOM_MODEL_CLASSES.append(auto_model_for_causal_lm_class)
        _MODEL_TYPE_TO_CAUSAL_LM_CLASS[model_type] = auto_model_for_causal_lm_class


def is_custom_model(model_type: str) -> bool:
    return model_type in _CUSTOM_MODEL_TYPES


def get_causal_lm_class(model_type: str) -> type[CausalLMModelMixin]:
    """returns the raw (non-HF-adapter-wrapped) lm_engine CausalLM class for a custom `model_type`, for callers
    (e.g. the training-only `ModelWrapper`) that must bypass `AutoModelForCausalLM`, which is registered to
    `LLMAdapter_HF` for HF compatibility"""

    assert is_custom_model(model_type), f"{model_type} is not a registered custom lm_engine model_type"
    return _MODEL_TYPE_TO_CAUSAL_LM_CLASS[model_type]
