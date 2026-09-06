# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch

from lm_engine.accelerator import KernelBackend
from lm_engine.enums import Kernel
from lm_engine.generation_cache import GenerationCache
from lm_engine.kernels import enable_kernels
from lm_engine.modeling_utils.sequence_mixer_blocks import RNN, RNNArgs, rnn
from tests.layers.utils import assert_equal_tensors, get_duplicated_tensors, skip_if_incompatible_kernel_backend


_HIDDEN_SIZE = 32
_BATCH = 2
_PREFILL_LEN = 16
_SEED = 42

# (state_head_dim, num_input_heads, num_weight_heads); num_input_heads != num_weight_heads
# exercises the grouped-head repeat_interleave broadcasting
_PROBLEM_SHAPES = [(8, 4, 8), (8, 8, 4), (9, 7, 7)]
_DTYPES = [torch.float32, torch.float16]


def _make_rnn(device: torch.device, dtype: torch.dtype, problem_shape: tuple[int, int, int]) -> RNN:
    state_head_dim, num_input_heads, num_weight_heads = problem_shape

    config = RNNArgs(
        state_head_dim=state_head_dim,
        num_input_heads=num_input_heads,
        num_weight_heads=num_weight_heads,
        add_bias=False,
        normalization_function="rmsnorm",
        gradient_clipping=None,
        kernel_size=4,
        activation_function="silu",
    )

    torch.manual_seed(42)
    rnn_module = RNN(
        input_size=_HIDDEN_SIZE,
        output_size=_HIDDEN_SIZE,
        config=config,
        initializer_range=0.02,
        m_width=1.0,
        init_method="normal",
        num_layers=1,
        layer_idx=0,
        use_depth_scaled_init=False,
        use_padding_free_transformer=False,
    ).to(device=device, dtype=dtype)
    rnn_module.eval()

    return rnn_module


@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_prefill_vs_torch_prefill(dtype: torch.dtype, problem_shape: tuple[int, int, int]) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    rnn_module = _make_rnn(device, dtype, problem_shape)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device, dtype=dtype)

    with enable_kernels([Kernel.rnn]):
        out_k = rnn_module(x)

    out_f = rnn_module(x)

    assert_equal_tensors(out_k, out_f, False)


@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_decode_vs_torch_decode(dtype: torch.dtype, problem_shape: tuple[int, int, int]) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    rnn_module = _make_rnn(device, dtype, problem_shape)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device, dtype=dtype)
    x_gen = torch.randn(_BATCH, 1, _HIDDEN_SIZE, device=device, dtype=dtype)

    with enable_kernels([Kernel.rnn]):
        cache_k = GenerationCache()
        rnn_module(x, cache_params=cache_k)
        out_gen_k = rnn_module(x_gen, cache_params=cache_k)

    cache_f = GenerationCache()
    rnn_module(x, cache_params=cache_f)
    out_gen_f = rnn_module(x_gen, cache_params=cache_f)

    assert_equal_tensors(out_gen_k, out_gen_f, False)


@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("has_input_state", [False, True])
def test_torch_prefill_continuation(
    dtype: torch.dtype, problem_shape: tuple[int, int, int], has_input_state: bool
) -> None:
    torch.manual_seed(0)

    state_head_dim, num_input_heads, num_weight_heads = problem_shape
    num_heads = max(num_input_heads, num_weight_heads)

    batch_size, sequence_length = 2, 10
    split = 4

    x = torch.randn(batch_size, sequence_length, num_input_heads, state_head_dim, dtype=dtype)
    W = torch.randn(num_weight_heads, state_head_dim, state_head_dim, dtype=dtype)
    h0 = torch.randn(batch_size, num_heads, state_head_dim, dtype=dtype) if has_input_state else None

    def _run(x, h0):
        return rnn(input=x, weight=W, input_state=h0, kernel_backend=KernelBackend.torch)

    full_output, full_state = _run(x, h0)

    first_output, first_state = _run(x[:, :split], h0)
    second_output, second_state = _run(x[:, split:], first_state)

    chunked_output = torch.cat([first_output, second_output], dim=1)

    assert_equal_tensors(chunked_output, full_output, False)
    assert_equal_tensors(second_state, full_state, False)


def _get_op_level_tensors(
    device: torch.device,
    dtype: torch.dtype,
    problem_shape: tuple[int, int, int],
    batch_size: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state_head_dim, num_input_heads, num_weight_heads = problem_shape

    x_kernel, x_torch = get_duplicated_tensors(
        (batch_size, sequence_length, num_input_heads, state_head_dim), device=device, dtype=dtype
    )

    W_kernel, W_torch = get_duplicated_tensors(
        (num_weight_heads, state_head_dim, state_head_dim), device=device, dtype=dtype
    )

    return x_kernel, x_torch, W_kernel, W_torch


@pytest.mark.parametrize("is_compiling", [False, True])
@pytest.mark.parametrize("has_input_state", [False, True])
@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_op_forward_kernel_vs_torch(
    dtype: torch.dtype, problem_shape: tuple[int, int, int], has_input_state: bool, is_compiling: bool
) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    state_head_dim, num_input_heads, num_weight_heads = problem_shape
    num_heads = max(num_input_heads, num_weight_heads)

    x_kernel, x_torch, W_kernel, W_torch = _get_op_level_tensors(
        device, dtype, problem_shape, batch_size=4, sequence_length=32
    )

    h0_kernel = h0_torch = None
    if has_input_state:
        h0_kernel, h0_torch = get_duplicated_tensors((4, num_heads, state_head_dim), device=device, dtype=dtype)

    rnn_fn = torch.compile(rnn, fullgraph=True) if is_compiling else rnn

    y_kernel, h_kernel = rnn_fn(
        input=x_kernel, weight=W_kernel, input_state=h0_kernel, kernel_backend=KernelBackend.triton
    )
    y_torch, h_torch = rnn(input=x_torch, weight=W_torch, input_state=h0_torch, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(y_kernel, y_torch, False)
    assert_equal_tensors(h_kernel, h_torch, False)


@pytest.mark.parametrize("is_compiling", [False, True])
@pytest.mark.parametrize("has_input_state", [False, True])
@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_op_backward_kernel_vs_torch(
    dtype: torch.dtype, problem_shape: tuple[int, int, int], has_input_state: bool, is_compiling: bool
) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    state_head_dim, num_input_heads, num_weight_heads = problem_shape
    num_heads = max(num_input_heads, num_weight_heads)

    x_kernel, x_torch, W_kernel, W_torch = _get_op_level_tensors(
        device, dtype, problem_shape, batch_size=4, sequence_length=32
    )

    h0_kernel = h0_torch = None
    if has_input_state:
        h0_kernel, h0_torch = get_duplicated_tensors((4, num_heads, state_head_dim), device=device, dtype=dtype)

    rnn_fn = torch.compile(rnn, fullgraph=True) if is_compiling else rnn

    y_kernel, _ = rnn_fn(input=x_kernel, weight=W_kernel, input_state=h0_kernel, kernel_backend=KernelBackend.triton)
    y_torch, _ = rnn(input=x_torch, weight=W_torch, input_state=h0_torch, kernel_backend=KernelBackend.torch)

    y_kernel.sum().backward()
    y_torch.sum().backward()

    assert_equal_tensors(y_kernel, y_torch, False)
    assert_equal_tensors(x_kernel.grad, x_torch.grad, False)
    assert_equal_tensors(W_kernel.grad, W_torch.grad, False)

    if has_input_state:
        assert_equal_tensors(h0_kernel.grad, h0_torch.grad, False)


@pytest.mark.parametrize("is_compiling", [False, True])
@pytest.mark.parametrize("has_input_state", [False, True])
@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_op_varlen_kernel_vs_torch(
    dtype: torch.dtype, problem_shape: tuple[int, int, int], has_input_state: bool, is_compiling: bool
) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    state_head_dim, num_input_heads, num_weight_heads = problem_shape
    num_heads = max(num_input_heads, num_weight_heads)

    cu_seqlens = torch.tensor([0, 7, 19, 27, 93], device=device)
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    B = cu_seqlens.size(0) - 1
    total_tokens = int(cu_seqlens[-1])

    x_kernel, x_torch = get_duplicated_tensors(
        (total_tokens, num_input_heads, state_head_dim), device=device, dtype=dtype
    )
    W_kernel, W_torch = get_duplicated_tensors(
        (num_weight_heads, state_head_dim, state_head_dim), device=device, dtype=dtype
    )

    h0_kernel = h0_torch = None
    if has_input_state:
        h0_kernel, h0_torch = get_duplicated_tensors((B, num_heads, state_head_dim), device=device, dtype=dtype)

    rnn_fn = torch.compile(rnn, fullgraph=True) if is_compiling else rnn

    y_kernel, h_kernel = rnn_fn(
        input=x_kernel,
        weight=W_kernel,
        input_state=h0_kernel,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=KernelBackend.triton,
    )

    y_torch = []
    h_torch = []
    for i in range(B):
        y, h = rnn(
            input=x_torch[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
            weight=W_torch,
            input_state=h0_torch[i].unsqueeze(0) if has_input_state else None,
            kernel_backend=KernelBackend.torch,
        )
        y_torch.append(y.squeeze(0))
        h_torch.append(h)

    y_torch = torch.cat(y_torch)
    h_torch = torch.cat(h_torch)

    assert_equal_tensors(y_kernel, y_torch, False)
    assert_equal_tensors(h_kernel, h_torch, False)
