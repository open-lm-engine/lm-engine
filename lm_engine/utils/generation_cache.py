# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any


_IS_GENERATION_CACHE_ENABLED: bool = True


class disable_generation_cache:
    def __enter__(self) -> Any:
        global _IS_GENERATION_CACHE_ENABLED
        self.original = _IS_GENERATION_CACHE_ENABLED

        _IS_GENERATION_CACHE_ENABLED = False

    def __exit__(self, exception_type, exception_value, exception_traceback) -> Any:
        global _IS_GENERATION_CACHE_ENABLED
        _IS_GENERATION_CACHE_ENABLED = self.original


def is_generation_cache_enabled() -> bool:
    return _IS_GENERATION_CACHE_ENABLED
