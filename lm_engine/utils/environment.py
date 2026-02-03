# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os
from contextlib import contextmanager


@contextmanager
def environment(env: dict):
    original_env = {}
    for key, value in env.items():
        original_env[key] = os.environ.get(key, "")
        os.environ[key] = value

    yield

    for key, value in original_env.items():
        os.environ[key] = value
