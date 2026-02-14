# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

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
        model_type = config_class.model_type

        AutoConfig.register(model_type, config_class)
        AutoModel.register(config_class, auto_model_class)
        AutoModelForCausalLM.register(config_class, auto_model_for_causal_lm_class)

        _CUSTOM_MODEL_TYPES.append(model_type)
        _CUSTOM_MODEL_CLASSES.append(auto_model_for_causal_lm_class)


def is_custom_model(model_type: str) -> bool:
    return model_type in _CUSTOM_MODEL_TYPES


_MODEL_PARALLEL_CLASS_MAPPING = {GPTBaseConfig.model_type: GPTBaseForCausalLM}


def get_model_parallel_class(model_type: str) -> AutoModelForCausalLM:
    if model_type in _MODEL_PARALLEL_CLASS_MAPPING:
        return _MODEL_PARALLEL_CLASS_MAPPING[model_type]

    raise ValueError(f"model parallelism is not supported with `model_type` ({model_type})")
