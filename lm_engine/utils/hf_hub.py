# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from transformers import AutoConfig, AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file
from transformers.utils.hub import get_checkpoint_shard_files


if TYPE_CHECKING:
    from ..tokenizers import TOKENIZER_TYPE


def download_repo(repo_name_or_path: str) -> tuple[AutoConfig | None, TOKENIZER_TYPE | None, str]:
    config = _download_config(repo_name_or_path)
    tokenizer = _download_tokenizer(repo_name_or_path)
    model_path = None

    if os.path.isdir(repo_name_or_path):
        model_path = repo_name_or_path
    else:
        # try downloading model weights
        try:
            model_path = cached_file(repo_name_or_path, SAFE_WEIGHTS_NAME)
            model_path = os.path.dirname(model_path)
        except:
            # try downloading model weights if they are sharded
            try:
                sharded_filename = cached_file(repo_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
                get_checkpoint_shard_files(repo_name_or_path, sharded_filename)
                model_path = os.path.dirname(sharded_filename)
            except:
                pass

    return config, tokenizer, model_path


def _download_config(repo_name_or_path: str) -> AutoConfig | None:
    try:
        config = AutoConfig.from_pretrained(repo_name_or_path)
    except:
        config = None

    return config


def _download_tokenizer(repo_name_or_path: str) -> TOKENIZER_TYPE | None:
    from ..tokenizers import get_tokenizer

    try:
        tokenizer = get_tokenizer(AutoTokenizer.__name__, repo_name_or_path)
    except:
        tokenizer = None

    return tokenizer
