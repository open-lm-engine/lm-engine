# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .send import _recv_op


def recv(x: torch.Tensor, shift: int = 1) -> None:
    """Blocking receive from rank `cp_rank - shift`, written into `x` in-place. Not differentiable."""
    _recv_op(y=x, shift=shift)
