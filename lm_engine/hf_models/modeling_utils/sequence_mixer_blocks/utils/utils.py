# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial
from typing import Callable

import torch

from .....enums import Kernel
from .....kernels import is_kernel_allowed
from .....utils import is_flash_attention_2_available, is_flash_attention_3_available, is_flash_attention_4_available
from .packing import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence


if is_flash_attention_2_available():
    from flash_attn import flash_attn_func as flash_attention_2
    from flash_attn import flash_attn_varlen_func as flash_attention_2_varlen
    from flash_attn.flash_attn_interface import _flash_attn_backward as _flash_attn_2_backward
    from flash_attn.flash_attn_interface import _flash_attn_forward as _flash_attn_2_forward

if is_flash_attention_3_available():
    from flash_attn_interface import _flash_attn_backward as _flash_attn_3_backward
    from flash_attn_interface import _flash_attn_forward as _flash_attn_3_forward
    from flash_attn_interface import flash_attn_func as flash_attention_3
    from flash_attn_interface import flash_attn_varlen_func as flash_attention_3_varlen

if is_flash_attention_4_available():
    from flash_attn.cute import flash_attn_func as flash_attention_4
    from flash_attn.cute import flash_attn_varlen_func as flash_attention_4_varlen


def _get_flash_attention_function(dropout: float) -> tuple[Callable, ...]:
    use_flash_attention_4 = is_kernel_allowed(Kernel.flash_attention_4)
    use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)
    use_flash_attention_2 = is_kernel_allowed(Kernel.flash_attention_2)

    assert (
        use_flash_attention_4 or use_flash_attention_3 or use_flash_attention_2
    ), "enable flash_attention_2, flash_attention_3, or flash_attention_4"

    if use_flash_attention_4 or use_flash_attention_3:
        assert dropout == 0

    if use_flash_attention_4:
        _flash_attention_function = flash_attention_4
        _flash_attention_varlen_function = flash_attention_4_varlen
    elif use_flash_attention_3:
        _flash_attention_function = flash_attention_3
        _flash_attention_varlen_function = flash_attention_3_varlen
        _flash_attention_forward = _flash_attn_3_forward
        _flash_attention_backward = _flash_attn_3_backward
    elif use_flash_attention_2:
        _flash_attention_function = partial(flash_attention_2, dropout_p=dropout)
        _flash_attention_varlen_function = partial(flash_attention_2_varlen, dropout_p=dropout)
        _flash_attention_forward = _flash_attn_2_forward
        _flash_attention_backward = _flash_attn_2_backward
    else:
        raise ValueError("unexpected flash_attention method")

    return (
        _flash_attention_function,
        _flash_attention_varlen_function,
        _flash_attention_forward,
        _flash_attention_backward,
    )


def _unpad_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cu_seqlens_k, max_seqlen_k = compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask)
    batch_size, kv_seq_len = key.size()[:2]

    if query_length == kv_seq_len:
        query, key, value = pack_sequence(inputs=(query, key, value), cu_seqlens=cu_seqlens_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_q = max_seqlen_k
    else:
        key, value = pack_sequence(inputs=(key, value), cu_seqlens=cu_seqlens_k)

        if query_length == 1:
            # There is a memcpy here, that is very bad.
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query.device)
            query = query.squeeze(1)
            max_seqlen_q = 1
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            cu_seqlens_q, max_seqlen_q = compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask)
            query = pack_sequence(inputs=query, cu_seqlens=cu_seqlens_q)

    return query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
