# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....custom_op import ctx_needs_gradients
from ....math import ceil_divide
from ..utils import _get_num_heads
from .forward import _linear_attention_forward_triton


class _LinearAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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

        B, S, _, K = k.size()
        V = v.size(-1)

        y = torch.empty(B, S, N, V, dtype=k.dtype, device=k.device)
        ht = torch.empty(B, N, K, V, dtype=torch.float32, device=k.device) if output_state else None

        BLOCK_SIZE_S = 128
        NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)

        h = (
            torch.empty(B, NUM_BLOCKS_S - 1, N, K, V, dtype=k.dtype, device=k.device)
            if ctx_needs_gradients(ctx)
            else None
        )

        _linear_attention_forward_triton(
            q=q,
            k=k,
            v=v,
            h0=h0,
            h=h,
            ht=ht,
            y=y,
            attention_multiplier=attention_multiplier,
            cu_seqlens=cu_seqlens,
            CHUNK_SIZE=BLOCK_SIZE_S,
        )

        return y, ht

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: torch.Tensor):
        raise NotImplementedError("backward is not implemented for kernel_backend (triton)")
