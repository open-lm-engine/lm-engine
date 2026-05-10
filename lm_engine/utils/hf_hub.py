# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import os

from transformers import AutoConfig, AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file
from transformers.utils.hub import get_checkpoint_shard_files


def download_repo(repo_name_or_path: str) -> tuple[AutoConfig | None, AutoTokenizer | None, str]:
    config = AutoConfig.from_pretrained(repo_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(repo_name_or_path)
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
