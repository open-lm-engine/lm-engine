# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os


def get_boolean_env_variable(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ["1", "true"]
