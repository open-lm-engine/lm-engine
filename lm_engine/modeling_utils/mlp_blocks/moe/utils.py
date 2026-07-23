# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from ....utils import is_triton_available, is_xma_available


if is_xma_available() and is_triton_available():
    from xma import continuous_count


# TODO add support for combileable bincount in PyTorch directly
@torch.library.custom_op("lm_engine::bincount", mutates_args={})
def bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return x.bincount(minlength=minlength).to(torch.uint32)


@bincount.register_fake
def _(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return torch.empty(minlength, device=x.device, dtype=torch.uint32)


def compute_bincount(x: torch.Tensor, size: int, use_continuous_count: bool) -> torch.Tensor:
    if use_continuous_count:
        count = continuous_count(x, bins=size)
    else:
        count = bincount(x, minlength=size)

    return count
