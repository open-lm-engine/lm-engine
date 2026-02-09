# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch

from .....enums import Kernel
from .....kernels import is_kernel_allowed
from .....utils import is_flash_attention_2_available, is_flash_attention_3_available
from .packing import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence


if is_flash_attention_2_available():
    from flash_attn import flash_attn_func as flash_attention_2
    from flash_attn import flash_attn_varlen_func as flash_attention_2_varlen

if is_flash_attention_3_available():
    from flash_attn_interface import flash_attn_func as flash_attention_3
    from flash_attn_interface import flash_attn_varlen_func as flash_attention_3_varlen


def unpad_input(
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


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
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
    use_flash_attention_2 = is_kernel_allowed(Kernel.flash_attention_2)
    use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)

    assert use_flash_attention_3 or use_flash_attention_2, "enable flash_attention_2 or flash_attention_3"

    if use_flash_attention_3:
        assert dropout == 0

        _flash_attention_function = flash_attention_3
        _flash_attention_varlen_function = flash_attention_3_varlen
    else:
        _flash_attention_function = partial(flash_attention_2, dropout_p=dropout)
        _flash_attention_varlen_function = partial(flash_attention_2_varlen, dropout_p=dropout)

    window_size = (-1, -1)
    if sliding_window is not None and key.size(1) > sliding_window:
        window_size = (sliding_window, sliding_window)

    kwargs = {"q": query, "k": key, "v": value, "softmax_scale": softmax_scale, "causal": causal}

    if use_padding_free_transformer:
        assert sliding_window is None

        kwargs.update(
            {
                "cu_seqlens_q": cu_seqlens,
                "cu_seqlens_k": cu_seqlens,
                "max_seqlen_q": max_seqlen,
                "max_seqlen_k": max_seqlen,
            }
        )

        attn_output = _flash_attention_varlen_function(**kwargs)
    elif attention_mask is None:
        kwargs.update({"window_size": window_size, "softcap": softcap})
        attn_output = _flash_attention_function(**kwargs)
    else:
        batch_size, query_length, num_heads, head_dim = query.size()

        query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = unpad_input(
            query, key, value, attention_mask, query_length
        )

        kwargs.update(
            {
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "window_size": window_size,
                "softcap": softcap,
            }
        )

        attn_output = _flash_attention_varlen_function(**kwargs)

        attn_output = unpack_sequence(
            inputs=attn_output, cu_seqlens=cu_seqlens_q, output_shape=(batch_size, query_length, num_heads, head_dim)
        )

    return attn_output
