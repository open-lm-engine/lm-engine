# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....autotuner import AutotuneConfig, autotune
from ....custom_op import xma_op
from ....math import ceil_divide
from ..utils import _get_num_heads
from .output_forward import _output_forward_triton_kernel
from .state_passing import _state_passing_forward_triton_kernel


@xma_op(mutates_args={"y", "h", "ht"})
@autotune(
    configs=[AutotuneConfig({"use_fused_kernel_in_forward": i}) for i in [True, False]],
    functional_triggers={
        "_": lambda **kwargs: (kwargs["q"].size(1) if kwargs["cu_seqlens"] is None else kwargs["max_seqlen"]) <= 64
    },
)
def _linear_attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    h: torch.Tensor,
    ht: torch.Tensor | None,
    y: torch.Tensor,
    attention_multiplier: float,
    cu_seqlens: torch.Tensor | None,
    CHUNK_SIZE: int,
    use_fused_kernel_in_forward: bool,
) -> None:
    Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)

    if cu_seqlens is None:
        B, S, _, K = k.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        K = k.size(-1)

    V = v.size(-1)

    kwargs = {
        "k_ptr": k,
        "k_stride": k.stride(),
        "v_ptr": v,
        "v_stride": v.stride(),
        "h0_ptr": h0,
        "h0_stride": None if h0 is None else h0.stride(),
        "h_ptr": h,
        "h_stride": None if h is None else h.stride(),
        "attention_multiplier": attention_multiplier,
        "cu_seqlens_ptr": cu_seqlens,
        "cu_seqlens_stride": None if cu_seqlens is None else cu_seqlens.stride(),
        "S": S,
        "N": N,
        "K": K,
        "V": V,
        "Gq": N // Nq,
        "Gk": N // Nk,
        "Gv": N // Nv,
    }

    GRID = lambda kwargs: (B * N, ceil_divide(K, kwargs["BLOCK_SIZE_K"]), ceil_divide(V, kwargs["BLOCK_SIZE_V"]))

    if use_fused_kernel_in_forward:
        _state_passing_forward_triton_kernel[GRID](
            q_ptr=q,
            q_stride=q.stride(),
            ht_ptr=ht,
            ht_stride=None if ht is None else ht.stride(),
            y_ptr=y,
            y_stride=y.stride(),
            CHUNK_SIZE=CHUNK_SIZE,
            **kwargs,
        )
    else:
        _state_passing_forward_triton_kernel[GRID](
            q_ptr=None,
            q_stride=None,
            ht_ptr=ht,
            ht_stride=None if ht is None else ht.stride(),
            y_ptr=None,
            y_stride=None,
            CHUNK_SIZE=CHUNK_SIZE,
            **kwargs,
        )

        NUM_CHUNKS = h.size(1)
        GRID = lambda kwargs: (B * N, NUM_CHUNKS + 1, ceil_divide(V, kwargs["BLOCK_SIZE_V"]))

        _output_forward_triton_kernel[GRID](
            q_ptr=q, q_stride=q.stride(), y_ptr=y, y_stride=y.stride(), BLOCK_SIZE_S=CHUNK_SIZE, **kwargs
        )
