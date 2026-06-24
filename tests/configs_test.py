# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import glob
import os

import pytest

from lm_engine.arguments import DistillationArgs, TrainingArgs, UnshardingArgs
from lm_engine.utils import load_yaml


_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
_ARGS_CLASS_BY_KEY = {"teacher_args": DistillationArgs, "unsharded_path": UnshardingArgs, "model_args": TrainingArgs}


def _collect_configs():
    paths = sorted(glob.glob(os.path.join(_CONFIGS_DIR, "**", "*.yml"), recursive=True))
    result = []
    for path in paths:
        d = load_yaml(path)
        for key, cls in _ARGS_CLASS_BY_KEY.items():
            if key in d:
                rel = os.path.relpath(path, _CONFIGS_DIR)
                result.append(pytest.param(path, cls, id=rel))
                break
    return result


@pytest.mark.parametrize("config_path,args_class", _collect_configs())
def test_config_is_valid(config_path: str, args_class: type) -> None:
    args_class(**load_yaml(config_path))
