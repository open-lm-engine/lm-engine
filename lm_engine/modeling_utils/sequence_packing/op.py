# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Sequence

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp
from ...utils import is_cute_dsl_available, is_triton_available
from .torch_implementation import _pack_torch, _unpack_torch


class _PackSequence(CustomOp): ...


class _UnpackSequence(CustomOp): ...


_PackSequence[KernelBackend.torch] = _pack_torch
_UnpackSequence[KernelBackend.torch] = _unpack_torch


if is_cute_dsl_available():
    from .cuda_implementation import _PackSequenceCUDA, _UnpackSequenceCUDA

    _PackSequence[KernelBackend.cuda] = _PackSequenceCUDA
    _UnpackSequence[KernelBackend.cuda] = _UnpackSequenceCUDA

if is_triton_available():
    from .triton_implementation import _PackSequenceTriton, _UnpackSequenceTriton

    _PackSequence[KernelBackend.triton] = _PackSequenceTriton
    _UnpackSequence[KernelBackend.triton] = _UnpackSequenceTriton


def pack_sequence(
    inputs: Sequence[torch.Tensor],
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    padding_side: str = "left",
    *,
    kernel_backend: KernelBackend | None = None,
) -> Sequence[torch.Tensor]:
    """
    pack tensors

    :param inputs: list of tensors
    :type inputs: Sequence[torch.Tensor]
    :param cu_seqlens: cumulative sequence length
    :type cu_seqlens: torch.Tensor
    :param total_tokens: total number of tokens
    :type total_tokens: int
    :param padding_side: padding side
    :type padding_side: str
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: list of packed tensors
    :rtype: Sequence[Tensor]
    """

    assert padding_side in ["left", "right"]
    assert isinstance(inputs, (list, tuple))

    outputs = []

    for x in inputs:
        assert x.dim() >= 2
        assert x.size(0) == cu_seqlens.size(0) - 1

        x = _PackSequence.run(
            x=x,
            cu_seqlens=cu_seqlens,
            output_shape=(total_tokens, *x.size()[2:]),
            padding_side=padding_side,
            kernel_backend=kernel_backend,
        )

        outputs.append(x)

    return outputs


def unpack_sequence(
    inputs: Sequence[torch.Tensor],
    cu_seqlens: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    padding_side: str = "left",
    *,
    kernel_backend: KernelBackend | None = None,
) -> Sequence[torch.Tensor]:
    """
    unpack tensors

    :param inputs: list of tensors
    :type inputs: Sequence[torch.Tensor]
    :param cu_seqlens: cumulative sequence length
    :type cu_seqlens: torch.Tensor
    :param batch_size: batch size
    :type batch_size: int
    :param sequence_length: sequence length
    :type sequence_length: int
    :param padding_side: padding side
    :type padding_side: str
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: list of unpacked tensors
    :rtype: Sequence[Tensor]
    """

    assert padding_side in ["left", "right"]
    assert isinstance(inputs, (list, tuple))

    outputs = []

    for x in inputs:
        assert x.dim() >= 2
        assert cu_seqlens.size(0) - 1 == batch_size

        x = _UnpackSequence.run(
            x=x,
            cu_seqlens=cu_seqlens,
            output_shape=(batch_size, sequence_length, *x.size()[1:]),
            padding_side=padding_side,
            kernel_backend=kernel_backend,
        )

        outputs.append(x)

    return outputs
