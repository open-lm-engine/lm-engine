# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp
from ...utils import is_triton_available
from .torch_implementation import _m2rnn_torch
from .utils import _get_num_heads


class _M2RNN(CustomOp): ...


_M2RNN[KernelBackend.torch] = _m2rnn_torch


if is_triton_available():
    from .triton_implementation import _M2RNNTriton

    _M2RNN[KernelBackend.cuda] = _M2RNNTriton
    _M2RNN[KernelBackend.triton] = _M2RNNTriton


def m2rnn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    computes M2RNN recurrence

    :param query: query tensor of shape (B, S, Nq, K) where Nq is the number of query heads and K is the key head
        dimension. Should have shape (T, Nq, K) and `cu_seqlens` should be passed.
    :type query: torch.Tensor
    :param key: key tensor of shape (B, S, Nk, K) where Nk is the number of key heads and K is the key head
        dimension. Should have shape (T, Nk, K) and `cu_seqlens` should be passed.
    :type key: torch.Tensor
    :param value: value tensor of shape (B, S, Nv, V) where Nv is the number of value heads and V is the value head
        dimension. Should have shape (T, Nv, V) and `cu_seqlens` should be passed.
    :type value: torch.Tensor
    :param weight: weight tensor of shape (Nw, V, V)
    :type weight: torch.Tensor
    :param forget_input: forget input tensor of shape (B, S, Nxf) where Nxf is the number of forget heads and H is the head
        dimension. Should have shape (T, Nxf) and `cu_seqlens` should be passed.
    :type forget_input: torch.Tensor
    :param input_state: starting state of shape (B, N, K, V), where N = max{Nq, Nk, Nv, Nxf, Nw}. None means starting state is
        0 tensor. Defaults to None.
    :type input_state: torch.Tensor | None
    :param gradient_clipping: gradient clipping for the state gradient in backward, None implies no clipping.
        Defaults to None.
    :type gradient_clipping: float | None
    :param cu_seqlens: cumulative sequence length (must contain 0 as first element). Defaults to None.
    :type cu_seqlens: torch.Tensor | None
    :param max_seqlen: max sequence length in the batch. Defaults to None.
    :type max_seqlen: int | None
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor of shape (B, S, N, V) if `cu_seqlens` is None else (T, N, V) and output state of
        shape (B, N, K, V).
    :rtype: tuple[Tensor, Tensor]
    """

    if cu_seqlens is None:
        assert max_seqlen is None
        B, S, _, K = query.size()
    else:
        assert max_seqlen is not None
        assert cu_seqlens.dim() == 1

        B = cu_seqlens.size(0) - 1
        T, _, K = query.size()

    V = value.size(-1)

    Nq, Nk, Nv, Nw, Nxf, N = _get_num_heads(q=query, k=key, v=value, W=weight, xf=forget_input, run_check=True)

    if cu_seqlens is None:
        assert query.size() == (B, S, Nq, K)
        assert key.size() == (B, S, Nk, K)
        assert value.size() == (B, S, Nv, V)
        assert forget_input.size() == (B, S, Nxf)
    else:
        assert query.size() == (T, Nq, K)
        assert key.size() == (T, Nk, K)
        assert value.size() == (T, Nv, V)
        assert forget_input.size() == (T, Nxf)

    assert weight.size() == (Nw, V, V)

    if input_state is not None:
        assert input_state.size() == (B, N, K, V)

    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    output, input_state = _M2RNN.run(
        q=query,
        k=key,
        v=value,
        W=weight,
        xf=forget_input,
        h0=input_state,
        gradient_clipping=gradient_clipping,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    return output, input_state
