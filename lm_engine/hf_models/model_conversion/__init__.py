# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import AutoConfig

from .bigcode import export_to_huggingface_bigcode, import_from_huggingface_bigcode
from .granite import export_to_huggingface_granite, import_from_huggingface_granite
from .granitemoe import export_to_huggingface_granitemoe, import_from_huggingface_granitemoe
from .granitemoeshared import export_to_huggingface_granitemoeshared, import_from_huggingface_granitemoeshared
from .granitemoesharedhybrid import export_to_huggingface_granitemoehybrid, import_from_huggingface_granitemoehybrid
from .llama import export_to_huggingface_llama, import_from_huggingface_llama


_MODEL_IMPORT_FUNCTIONS = {
    "gpt_bigcode": import_from_huggingface_bigcode,
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
    import_function(pretrained_model_name_or_path, save_path)


_MODEL_EXPORT_FUNCTIONS = {
    "gpt_bigcode": export_to_huggingface_bigcode,
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
