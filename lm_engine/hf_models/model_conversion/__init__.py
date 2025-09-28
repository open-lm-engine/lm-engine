# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import AutoConfig, GenerationConfig

from ...utils import SafeTensorsWeightsManager
from .granite import export_to_huggingface_granite, import_from_huggingface_granite
from .granitemoe import export_to_huggingface_granitemoe, import_from_huggingface_granitemoe
from .granitemoehybrid import export_to_huggingface_granitemoehybrid, import_from_huggingface_granitemoehybrid
from .granitemoeshared import export_to_huggingface_granitemoeshared, import_from_huggingface_granitemoeshared
from .llama import export_to_huggingface_llama, import_from_huggingface_llama


_MODEL_IMPORT_FUNCTIONS = {
    "granite": import_from_huggingface_granite,
    "granitemoe": import_from_huggingface_granitemoe,
    "granitemoeshared": import_from_huggingface_granitemoeshared,
    "granitemoehybrid": import_from_huggingface_granitemoehybrid,
    "llama": import_from_huggingface_llama,
}


def import_from_huggingface(pretrained_model_name_or_path: str, save_path: str) -> None:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    model_type = config.model_type

    if model_type not in _MODEL_IMPORT_FUNCTIONS:
        raise NotImplementedError(f"the current model_type ({model_type}) is not yet supported")

    import_function = _MODEL_IMPORT_FUNCTIONS[model_type]

    config, tokenizer, state_dict = import_function(pretrained_model_name_or_path)
    generation_config = GenerationConfig.from_model_config(config)

    config.save_pretrained(save_path)
    generation_config.save_pretrained(save_path)

    SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path, legacy_format=False)


_MODEL_EXPORT_FUNCTIONS = {
    "granite": export_to_huggingface_granite,
    "granitemoe": export_to_huggingface_granitemoe,
    "granitemoeshared": export_to_huggingface_granitemoeshared,
    "granitemoehybrid": export_to_huggingface_granitemoehybrid,
    "llama": export_to_huggingface_llama,
}


def export_to_huggingface(pretrained_model_name_or_path: str, save_path: str, model_type: str) -> None:
    if model_type not in _MODEL_EXPORT_FUNCTIONS:
        raise NotImplementedError(f"the current model_type ({model_type}) is not yet supported")

    export_function = _MODEL_EXPORT_FUNCTIONS[model_type]
    export_function(pretrained_model_name_or_path, save_path)
