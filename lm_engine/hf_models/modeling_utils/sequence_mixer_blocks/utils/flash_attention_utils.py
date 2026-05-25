# **************************************************
# Copyright (c) 2026, Mayank Mishra, Bharat
# **************************************************

import torch

from .....enums import Kernel
from .....kernels import is_kernel_allowed
from .....parallel import ProcessGroupManager
from .packing import unpack_sequence
from .ring_attention import ring_attention_function
from .utils import _get_flash_attention_function, _unpad_input


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
    use_padding_free_transformer: bool,
    causal: bool,
    dropout: float = 0,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    softcap: float = 0,
) -> torch.Tensor:
    window_size = (-1, -1)
    if sliding_window is not None and k.size(1) > sliding_window:
        window_size = (sliding_window, sliding_window)

    use_flash_attention_4 = is_kernel_allowed(Kernel.flash_attention_4)
    (
        _flash_attention_function,
        _flash_attention_varlen_function,
        _flash_attention_forward,
        _flash_attention_backward,
    ) = _get_flash_attention_function(dropout=dropout)

    if ProcessGroupManager.is_context_parallel_enabled():
        assert dropout == 0

        x = ring_attention_function(
            q=q,
            k=k,
            v=v,
            causal=causal,
            softmax_scale=softmax_scale,
            window_size=window_size,
            softcap=softcap,
            forward_function=_flash_attention_forward,
            backward_function=_flash_attention_backward,
        )
    else:
        if use_padding_free_transformer:
            assert not ProcessGroupManager.is_context_parallel_enabled()
            assert sliding_window is None

            x = _flash_attention_varlen_function(
                q=q,
                k=k,
                v=v,
                softmax_scale=softmax_scale,
                causal=causal,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                window_size=window_size,
                softcap=softcap,
            )

            if use_flash_attention_4:
                assert isinstance(x, tuple)
                x = x[0]
        elif attention_mask is None:
            x = _flash_attention_function(
                q=q,
                k=k,
                v=v,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
            )

            if use_flash_attention_4:
                assert isinstance(x, tuple)
                x = x[0]
        else:
            B, S, N, H = q.size()
            assert not ProcessGroupManager.is_context_parallel_enabled()

            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = _unpad_input(q, k, v, attention_mask, S)

            x = _flash_attention_varlen_function(
                q=q,
                k=k,
                v=v,
                softmax_scale=softmax_scale,
                causal=causal,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                window_size=window_size,
                softcap=softcap,
            )

            if use_flash_attention_4:
                assert isinstance(x, tuple)
                x = x[0]

            x = unpack_sequence(inputs=x, cu_seqlens=cu_seqlens_q, output_shape=(B, S, N, H))

    return x
