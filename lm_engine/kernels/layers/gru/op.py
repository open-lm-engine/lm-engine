# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp
from ...utils import is_triton_available
from .torch_implementation import _gru_torch
from .utils import _get_num_heads


class _GRU(CustomOp): ...


_GRU[KernelBackend.torch] = _gru_torch


if is_triton_available():
    from .triton_implementation import _GRUTriton

    _GRU[KernelBackend.cuda] = _GRUTriton
    _GRU[KernelBackend.triton] = _GRUTriton


def gru(
    input: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    forget_weight: torch.Tensor,
    reset_input: torch.Tensor,
    reset_weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    computes multihead RNN: `tanh(input_state @ weight + input)`

    :param input: input tensor of shape (B, S, Nx, H) where Nx is the number of input heads and H is the head
        dimension. Should have shape (T, Nx, H) and `cu_seqlens` should be passed.
    :type input: torch.Tensor
    :param weight: weight tensor of shape (Nw, H, H)
    :type weight: torch.Tensor
    :param forget_input: forget input tensor of shape (B, S, Nxf, H) where Nxf is the number of input heads and H is the head
        dimension. Should have shape (T, Nxf, H) and `cu_seqlens` should be passed.
    :type forget_input: torch.Tensor
    :param forget_weight: forget weight tensor of shape (NWf, H, H)
    :type forget_weight: torch.Tensor
    :param reset_input: reset input tensor of shape (B, S, Nxr, H) where Nxr is the number of input heads and H is the head
        dimension. Should have shape (T, Nxr, H) and `cu_seqlens` should be passed.
    :type reset_input: torch.Tensor
    :param reset_weight: reset weight tensor of shape (Nwr, H, H)
    :type reset_weight: torch.Tensor
    :param input_state: starting state of shape (B, N, H), where N = max{Nx, Nw}. None means starting state is
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
    :return: output tensor of shape (B, S, N, H) if `cu_seqlens` is None else (T, N, H) and output state of
        shape (B, N, H).
    :rtype: tuple[Tensor, Tensor]
    """

    if cu_seqlens is None:
        assert max_seqlen is None
        B, S, _, H = input.size()
    else:
        assert max_seqlen is not None
        assert cu_seqlens.dim() == 1

        B = cu_seqlens.size(0) - 1
        T, _, H = input.size()

    Nx, Nxf, Nxr, Nw, Nwf, Nwr, N = _get_num_heads(
        x=input, W=weight, xf=forget_input, Wf=forget_weight, xr=reset_input, Wr=reset_weight, run_check=True
    )

    if cu_seqlens is None:
        input.size() == (B, S, Nx, H)
        forget_input.size() == (B, S, Nxf, H)
        reset_input.size() == (B, S, Nxr, H)
    else:
        input.size() == (T, Nx, H)
        forget_input.size() == (T, Nxf, H)
        reset_input.size() == (T, Nxr, H)

    assert weight.size() == (Nw, H, H)
    assert forget_weight.size() == (Nwf, H, H)
    assert reset_weight.size() == (Nwr, H, H)

    if input_state is not None:
        assert input_state.size() == (B, N, H)

    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    input, input_state = _GRU.run(
        x=input,
        W=weight,
        xf=forget_input,
        Wf=forget_weight,
        xr=reset_input,
        Wr=reset_weight,
        h0=input_state,
        gradient_clipping=gradient_clipping,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    return input, input_state
