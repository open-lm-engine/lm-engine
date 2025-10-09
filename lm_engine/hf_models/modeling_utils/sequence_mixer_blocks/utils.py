# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import is_flash_attention_2_available, is_flash_attention_3_available
from ...mask import AttentionMaskInfo


if is_flash_attention_2_available():
    from flash_attn import flash_attn_func as flash_attention_2
    from flash_attn import flash_attn_varlen_func as flash_attention_2_varlen

if is_flash_attention_3_available():
    from flash_attn_interface import flash_attn_func as flash_attention_3
    from flash_attn_interface import flash_attn_varlen_func as flash_attention_3_varlen


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask_info: AttentionMaskInfo,
    causal: bool,
    dropout: float = 0,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    softcap: float = 0,
) -> torch.Tensor:
    use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)

    if use_flash_attention_3:
        assert dropout == 0

    window_size = (-1, -1)
    if sliding_window is not None and k.size(1) > sliding_window:
        window_size = (sliding_window, sliding_window)

    assert q.dim() == 3
    assert k.dim() == 3
    assert v.dim() == 3

    if attention_mask_info.is_ragged():
        assert sliding_window is None

        cu_seqlens = attention_mask_info.get_cu_seqlens()
        max_seqlen = attention_mask_info.get_max_seqlen()

        if use_flash_attention_3:
            x, _ = flash_attention_3_varlen(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        else:
            x = flash_attention_2_varlen(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
    else:
        q, k, v = attention_mask_info.unpack_sequence((q, k, v))

        if use_flash_attention_3:
            x, _ = flash_attention_3(
                q=q,
                k=k,
                v=v,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
            )
        else:
            x = flash_attention_2(
                q=q,
                k=k,
                v=v,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
            )

        x = attention_mask_info.pack_sequence(x)

    return x
