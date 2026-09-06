# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch


def _get_num_heads(x: torch.Tensor, W: torch.Tensor, run_check: bool) -> tuple[int, int, int]:
    Nx = x.size(-2)
    Nw = W.size(0)
    N = max(Nx, Nw)

    if run_check:
        assert N % Nx == 0
        assert N % Nw == 0

    return Nx, Nw, N
