# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from ...utils import SafeTensorsWeightsManager


def save_config_tokenizer_model(
    config: AutoConfig, generation_config: GenerationConfig, tokenizer: AutoTokenizer, state_dict: dict, save_path: str
) -> None:
    config.save_pretrained(save_path)
    generation_config.save_pretrained(save_path)

    SafeTensorsWeightsManager.save_state_dict(state_dict, save_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path, legacy_format=False)
