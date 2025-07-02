# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from transformers import AutoTokenizer

from .alpha_numeric import AlphaNumericTokenizer


_CUSTOM_TOKENIZER_CLASS_MAP = {AlphaNumericTokenizer.__name__: AlphaNumericTokenizer}
TOKENIZER_TYPE = AutoTokenizer | AlphaNumericTokenizer


def get_tokenizer(tokenizer_class_name: str, tokenizer_name: str, **tokenizer_class_args: dict) -> TOKENIZER_TYPE:
    if tokenizer_class_name == AutoTokenizer.__name__:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        _CUSTOM_TOKENIZER_CLASS_MAP[tokenizer_class_name](**tokenizer_class_args)

    return tokenizer
