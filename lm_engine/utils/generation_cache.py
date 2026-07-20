# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from contextlib import contextmanager


_IS_GENERATION_CACHE_ENABLED: bool = True


@contextmanager
def disable_generation_cache():
    global _IS_GENERATION_CACHE_ENABLED
    original = _IS_GENERATION_CACHE_ENABLED

    _IS_GENERATION_CACHE_ENABLED = False

    yield

    _IS_GENERATION_CACHE_ENABLED = original


def is_generation_cache_enabled() -> bool:
    return _IS_GENERATION_CACHE_ENABLED
