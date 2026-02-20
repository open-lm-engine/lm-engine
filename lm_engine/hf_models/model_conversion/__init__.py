# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import AutoTokenizer, GenerationConfig

from ...tokenizers import get_tokenizer
from ...utils import SafeTensorsWeightsManager, download_repo
from ..models import GPTBaseConfig
from .granite import _export_granite_config, _import_granite_config
from .granitemoehybrid import (
    _export_granitemoehybrid_config,
    _export_granitemoehybrid_state_dict,
    _import_granitemoehybrid_config,
    _import_granitemoehybrid_state_dict,
)
from .llama import _export_llama_config, _export_llama_state_dict, _import_llama_config, _import_llama_state_dict


_MODEL_IMPORT_FUNCTIONS = {
    "granite": (_import_granite_config, _import_llama_state_dict),
    "granitemoehybrid": (_import_granitemoehybrid_config, _import_granitemoehybrid_state_dict),
    "llama": (_import_llama_config, _import_llama_state_dict),
}


def import_from_huggingface(
    pretrained_model_name_or_path: str, save_path: str | None = None
) -> tuple[GPTBaseConfig, GenerationConfig, AutoTokenizer, dict]:
    original_config, tokenizer, downloaded_model_path = download_repo(pretrained_model_name_or_path)
    model_type = original_config.model_type

    if model_type not in _MODEL_IMPORT_FUNCTIONS:
        raise NotImplementedError(f"the current model_type ({model_type}) is not yet supported")

    config_import_function, state_dict_import_function = _MODEL_IMPORT_FUNCTIONS[model_type]

    config = config_import_function(original_config)

    state_dict = state_dict_import_function(
        config=config, safetensors_weights_manager=SafeTensorsWeightsManager(downloaded_model_path)
    )

    generation_config = GenerationConfig.from_model_config(config)

    if save_path is not None:
        config.save_pretrained(save_path)
        generation_config.save_pretrained(save_path)

        SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)

        if tokenizer is not None:
            tokenizer.save_pretrained(save_path, legacy_format=False)

    return config, generation_config, tokenizer, state_dict


_MODEL_EXPORT_FUNCTIONS = {
    "granite": (_export_granite_config, _export_llama_state_dict),
    "granitemoehybrid": (_export_granitemoehybrid_config, _export_granitemoehybrid_state_dict),
    "llama": (_export_llama_config, _export_llama_state_dict),
}


def export_to_huggingface(
    pretrained_model_name_or_path: str, model_type: str, save_path: str | None = None
) -> tuple[GPTBaseConfig, GenerationConfig, AutoTokenizer, dict]:
    config, tokenizer, downloaded_model_path = download_repo(pretrained_model_name_or_path)

    if model_type not in _MODEL_EXPORT_FUNCTIONS:
        raise NotImplementedError(f"the current model_type ({model_type}) is not yet supported")

    config_export_function, state_dict_export_function = _MODEL_EXPORT_FUNCTIONS[model_type]

    state_dict = state_dict_export_function(
        config, safetensors_weights_manager=SafeTensorsWeightsManager(downloaded_model_path)
    )

    config = config_export_function(config)
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
