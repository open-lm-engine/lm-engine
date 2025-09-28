# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from ...tokenizers import get_tokenizer
from ...utils import SafeTensorsWeightsManager
from ..models import GPTBaseConfig
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


def import_from_huggingface(
    pretrained_model_name_or_path: str, save_path: str | None = None
) -> tuple[GPTBaseConfig, GenerationConfig, AutoTokenizer, dict]:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    model_type = config.model_type

    if model_type not in _MODEL_IMPORT_FUNCTIONS:
        raise NotImplementedError(f"the current model_type ({model_type}) is not yet supported")

    import_function = _MODEL_IMPORT_FUNCTIONS[model_type]

    config, tokenizer, state_dict = import_function(pretrained_model_name_or_path)
    generation_config = GenerationConfig.from_model_config(config)

    if save_path is not None:
        config.save_pretrained(save_path)
        generation_config.save_pretrained(save_path)

        SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)

        if tokenizer is not None:
            tokenizer.save_pretrained(save_path, legacy_format=False)

    return config, generation_config, tokenizer, state_dict


_MODEL_EXPORT_FUNCTIONS = {
    "granite": export_to_huggingface_granite,
    "granitemoe": export_to_huggingface_granitemoe,
    "granitemoeshared": export_to_huggingface_granitemoeshared,
    "granitemoehybrid": export_to_huggingface_granitemoehybrid,
    "llama": export_to_huggingface_llama,
}


def export_to_huggingface(
    pretrained_model_name_or_path: str, model_type: str, save_path: str | None = None
) -> tuple[GPTBaseConfig, GenerationConfig, AutoTokenizer, dict]:
    if model_type not in _MODEL_EXPORT_FUNCTIONS:
        raise NotImplementedError(f"the current model_type ({model_type}) is not yet supported")

    export_function = _MODEL_EXPORT_FUNCTIONS[model_type]

    config, state_dict = export_function(pretrained_model_name_or_path)
    generation_config = GenerationConfig.from_model_config(config)

    try:
        tokenizer = get_tokenizer(AutoTokenizer.__name__, pretrained_model_name_or_path)
    except:
        tokenizer = None

    if save_path is not None:
        config.save_pretrained(save_path)
        generation_config.save_pretrained(save_path)

        SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)

        if tokenizer is not None:
            tokenizer.save_pretrained(save_path, legacy_format=False)

    return config, generation_config, tokenizer, state_dict
