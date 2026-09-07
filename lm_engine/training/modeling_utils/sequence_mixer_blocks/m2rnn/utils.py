# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch


def _get_num_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    run_check: bool,
) -> tuple[int, int, int, int, int, int]:
    Nq = q.size(-2)
    Nk = k.size(-2)
    Nv = v.size(-2)

    Nw = W.size(0)
    Nxf = xf.size(-1)

    N = max(Nq, Nk, Nv, Nw, Nxf)

    if run_check:
        assert N % Nq == 0
        assert N % Nk == 0
        assert N % Nv == 0

        assert N % Nw == 0
        assert N % Nxf == 0

    return Nq, Nk, Nv, Nw, Nxf, N
