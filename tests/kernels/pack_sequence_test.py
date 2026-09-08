# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Callable

import pytest
import torch

from lm_engine.kernels import KernelBackend
from lm_engine.kernels.functional import pack_sequence, unpack_sequence
from tests.layers.utils import assert_equal_tensors, get_duplicated_tensors, skip_if_incompatible_kernel_backend


def _get_problem_shapes(packed: bool) -> list[tuple[int]]:
    sizes = []
    for i, j in [(14, 16), (12, 14), (12, 21), (9, 7), (1, 1), (11, 11), (19, 19)]:
        if packed:
            sizes.append((691, i, j))
        else:
            sizes.append((7, 1000, i, j))

    return sizes


@pytest.mark.parametrize("size", _get_problem_shapes(False))
@pytest.mark.parametrize("cu_seqlens", [[0, 70, 170, 295, 393, 412, 515, 691]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda, KernelBackend.triton])
@pytest.mark.parametrize("function", [pack_sequence, torch.compile(pack_sequence, fullgraph=True)])
@torch._dynamo.config.patch(recompile_limit=1024)
def test_pack_sequence(
    size: tuple[int],
    cu_seqlens: list[int],
    dtype: torch.dtype,
    padding_side: str,
    kernel_backend: KernelBackend,
    function: Callable,
) -> None:
    device = skip_if_incompatible_kernel_backend(kernel_backend)

    x_kernel, x_expected = get_duplicated_tensors(size, device=device, dtype=dtype, std=None)
    cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.uint32)

    z_kernel = function(
        [x_kernel],
        cu_seqlens=cu_seqlens,
        total_tokens=cu_seqlens[-1].item(),
        padding_side=padding_side,
        kernel_backend=kernel_backend,
    )[0]

    z_expected = pack_sequence(
        [x_expected],
        cu_seqlens=cu_seqlens.to(torch.int),
        total_tokens=cu_seqlens[-1].item(),
        padding_side=padding_side,
        kernel_backend=KernelBackend.torch,
    )[0]

    z_expected.sum().backward()
    z_kernel.sum().backward()

    assert_equal_tensors(z_kernel, z_expected, True)
    assert_equal_tensors(x_kernel.grad, x_expected.grad, True)


@pytest.mark.parametrize("size", _get_problem_shapes(True))
@pytest.mark.parametrize("cu_seqlens", [[0, 70, 170, 295, 393, 412, 515, 691]])
@pytest.mark.parametrize("sequence_length", [1000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda, KernelBackend.triton])
@pytest.mark.parametrize("function", [unpack_sequence, torch.compile(unpack_sequence, fullgraph=True)])
@torch._dynamo.config.patch(recompile_limit=1024)
def test_unpack_sequence(
    size: tuple[int],
    cu_seqlens: list[int],
    sequence_length: tuple[int],
    dtype: torch.dtype,
    padding_side: str,
    kernel_backend: KernelBackend,
    function: Callable,
) -> None:
    device = skip_if_incompatible_kernel_backend(kernel_backend)

    x_kernel, x_expected = get_duplicated_tensors(size, device=device, dtype=dtype, std=None)
    cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.uint32)

    z_kernel = function(
        [x_kernel],
        cu_seqlens=cu_seqlens,
        batch_size=cu_seqlens.size(0) - 1,
        sequence_length=sequence_length,
        padding_side=padding_side,
        kernel_backend=kernel_backend,
    )[0]

    z_expected = unpack_sequence(
        [x_expected],
        cu_seqlens=cu_seqlens.to(torch.int),
        batch_size=cu_seqlens.size(0) - 1,
        sequence_length=sequence_length,
        padding_side=padding_side,
        kernel_backend=KernelBackend.torch,
    )[0]

    z_expected.sum().backward()
    z_kernel.sum().backward()

    assert_equal_tensors(z_kernel, z_expected, True)
    assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
