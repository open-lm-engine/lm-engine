# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import math

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp
from ...utils import is_triton_available
from .torch_implementation import _linear_attention_torch
from .utils import _get_num_heads


class _LinearAttention(CustomOp): ...


_LinearAttention[KernelBackend.torch] = _linear_attention_torch


if is_triton_available():
    from .triton_implementation import _LinearAttentionTriton

    _LinearAttention[KernelBackend.cuda] = _LinearAttentionTriton
    _LinearAttention[KernelBackend.triton] = _LinearAttentionTriton


def linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    input_state: torch.Tensor | None,
    attention_multiplier: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    output_state: bool = False,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """computes linear attention: `y[s] = q[s] @ h[s]`, `h[s] = h[s - 1] + k[s].T @ v[s]`

    :param query: query tensor of shape (B, S, Nq, K). Should have shape (T, Nq, K) and `cu_seqlens` should be
        passed.
    :type query: torch.Tensor
    :param key: key tensor of shape (B, S, Nk, K). Should have shape (T, Nk, K) and `cu_seqlens` should be
        passed.
    :type key: torch.Tensor
    :param value: value tensor of shape (B, S, Nv, V). Should have shape (T, Nv, V) and `cu_seqlens` should be
        passed.
    :type value: torch.Tensor
    :param input_state: starting state of shape (B, N, K, V), where N = max{Nq, Nk, Nv}. None means starting
        state is 0 tensor.
    :type input_state: torch.Tensor | None
    :param attention_multiplier: scaling factor applied to the output, `y`. None defaults to `1 / sqrt(K)`.
        Defaults to None.
    :type attention_multiplier: float | None
    :param cu_seqlens: cumulative sequence length (must contain 0 as first element). Defaults to None.
    :type cu_seqlens: torch.Tensor | None
    :param max_seqlen: max sequence length in the batch. Defaults to None.
    :type max_seqlen: int | None
    :param output_state: whether to also return the final state (of shape (B, N, K, V)) for use as
        `input_state` in a subsequent call. Defaults to False.
    :type output_state: bool
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor of shape (B, S, N, V) if `cu_seqlens` is None else (T, N, V), and output state of
        shape (B, N, K, V) if `output_state` is True else None.
    :rtype: tuple[torch.Tensor, torch.Tensor | None]
    """

    if cu_seqlens is None:
        assert max_seqlen is None
        B, S, _, K = query.size()
    else:
        assert max_seqlen is not None
        assert cu_seqlens.dim() == 1

        T, _, K = query.size()
        B = cu_seqlens.size(0) - 1

    V = value.size(-1)
    Nq, Nk, Nv, N = _get_num_heads(q=query, k=key, v=value, run_check=True)

    if cu_seqlens is None:
        assert query.size() == (B, S, Nq, K)
        assert key.size() == (B, S, Nk, K)
        assert value.size() == (B, S, Nv, V)
    else:
        assert query.size() == (T, Nq, K)
        assert key.size() == (T, Nk, K)
        assert value.size() == (T, Nv, V)

    if input_state is not None:
        assert input_state.size() == (B, N, K, V)

    if attention_multiplier is None:
        attention_multiplier = 1 / math.sqrt(K)

    output, input_state = _LinearAttention.run(
        q=query,
        k=key,
        v=value,
        h0=input_state,
        attention_multiplier=attention_multiplier,
        output_state=output_state,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    return output, input_state
