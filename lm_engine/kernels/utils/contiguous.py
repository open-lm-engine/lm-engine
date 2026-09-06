# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any, Callable

import torch
from torch.utils._pytree import tree_map


def _make_contiguous(x: Any) -> Any:
    return x.contiguous() if isinstance(x, torch.Tensor) else x


def ensure_contiguous(func: Callable) -> Callable:
    def inner(*args, **kwargs):
        args = tree_map(_make_contiguous, args)
        kwargs = tree_map(_make_contiguous, kwargs)
        return func(*args, **kwargs)

    return inner
