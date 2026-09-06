# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .utils import _get_num_heads


def _linear_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    attention_multiplier: float,
    output_state: bool,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)

    y_shape = list(v.size())
    y_shape[-2] = N
    y = torch.empty(y_shape, device=q.device, dtype=q.dtype)

    if cu_seqlens is None:
        B, S, _, K = q.size()
    else:
        raise NotImplementedError

    V = v.size(-1)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    q = q.repeat_interleave(Gq, dim=-2)
    k = k.repeat_interleave(Gk, dim=-2)
    v = v.repeat_interleave(Gv, dim=-2)

    h0 = torch.zeros(B, N, K, V, dtype=torch.float32, device=q.device) if h0 is None else h0.float()

    for s in range(S):
        h0 = h0 + k[:, s, ..., None] * v[:, s, :, None, :]
        y[:, s] = (q[:, s, :, None, :] @ h0.type_as(q)).squeeze(-2)

    y = y * attention_multiplier

    if not output_state:
        h0 = None

    return y, h0
