# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch

from lm_engine.accelerator import KernelBackend
from lm_engine.enums import Kernel
from lm_engine.generation_cache import GenerationCache
from lm_engine.kernels import enable_kernels
from lm_engine.modeling_utils.sequence_mixer_blocks import M2RNN, M2RNNArgs, m2rnn
from tests.layers.utils import assert_equal_tensors, get_duplicated_tensors, skip_if_incompatible_kernel_backend


_HIDDEN_SIZE = 32
_BATCH = 2
_PREFILL_LEN = 16
_SEED = 42
_DTYPES = [torch.float32, torch.bfloat16]


def _get_problem_shapes() -> list[tuple[int, int, int, int, int, int, int]]:
    # (key_head_dim, value_head_dim, num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads);
    # each shape perturbs a single head-count/head-dim away from a common base, one at a time, so every
    # dimension gets exercised through the grouped-head repeat_interleave broadcasting at least once
    base = [64, 64, 8, 8, 8, 8, 8]

    result = [(9, 9, 7, 7, 7, 7, 7)]
    for i in range(1, len(base)):
        t = base.copy()
        t[i] = 4
        result.append(tuple(t))
    result.append((128, 64, 8, 8, 8, 8, 8))

    return result


_PROBLEM_SHAPES = _get_problem_shapes()


def _make_m2rnn(
    device: torch.device, dtype: torch.dtype, problem_shape: tuple[int, int, int, int, int, int, int]
) -> M2RNN:
    key_head_dim, value_head_dim, num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads = problem_shape

    config = M2RNNArgs(
        k_head_dim=key_head_dim,
        v_head_dim=value_head_dim,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        num_f_heads=num_f_heads,
        num_g_heads=num_v_heads,
        num_weight_heads=num_weight_heads,
        use_residual=True,
        kernel_size=4,
        activation_function="silu",
        add_bias=False,
        gradient_clipping=None,
        normalization_function="rmsnorm",
        A_init_min=1,
        A_init_max=16,
        dt_init_min=0.001,
        dt_init_max=0.1,
        dt_init_floor=1e-4,
    )

    torch.manual_seed(42)
    m2rnn_module = M2RNN(
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
    m2rnn_module.eval()

    return m2rnn_module


@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_prefill_vs_torch_prefill(
    dtype: torch.dtype, problem_shape: tuple[int, int, int, int, int, int, int]
) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    m2rnn_module = _make_m2rnn(device, dtype, problem_shape)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device, dtype=dtype)

    with enable_kernels([Kernel.m2rnn]):
        out_k = m2rnn_module(x)

    out_f = m2rnn_module(x)

    assert_equal_tensors(out_k, out_f, False)


@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_decode_vs_torch_decode(
    dtype: torch.dtype, problem_shape: tuple[int, int, int, int, int, int, int]
) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    m2rnn_module = _make_m2rnn(device, dtype, problem_shape)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device, dtype=dtype)
    x_gen = torch.randn(_BATCH, 1, _HIDDEN_SIZE, device=device, dtype=dtype)

    with enable_kernels([Kernel.m2rnn]):
        cache_k = GenerationCache()
        m2rnn_module(x, cache_params=cache_k)
        out_gen_k = m2rnn_module(x_gen, cache_params=cache_k)

    cache_f = GenerationCache()
    m2rnn_module(x, cache_params=cache_f)
    out_gen_f = m2rnn_module(x_gen, cache_params=cache_f)

    assert_equal_tensors(out_gen_k, out_gen_f, False)


@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("has_input_state", [False, True])
def test_torch_prefill_continuation(
    dtype: torch.dtype, problem_shape: tuple[int, int, int, int, int, int, int], has_input_state: bool
) -> None:
    torch.manual_seed(0)

    key_head_dim, value_head_dim, num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads = problem_shape
    num_heads = max(num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads)

    batch_size, sequence_length = 2, 10
    split = 4

    q = torch.randn(batch_size, sequence_length, num_q_heads, key_head_dim, dtype=dtype)
    k = torch.randn(batch_size, sequence_length, num_k_heads, key_head_dim, dtype=dtype)
    v = torch.randn(batch_size, sequence_length, num_v_heads, value_head_dim, dtype=dtype)
    xf = torch.rand(batch_size, sequence_length, num_f_heads, dtype=dtype)
    W = torch.randn(num_weight_heads, value_head_dim, value_head_dim, dtype=dtype)
    h0 = torch.randn(batch_size, num_heads, key_head_dim, value_head_dim, dtype=dtype) if has_input_state else None

    def _run(q, k, v, xf, h0):
        return m2rnn(
            query=q,
            key=k,
            value=v,
            weight=W,
            forget_input=xf,
            input_state=h0,
            kernel_backend=KernelBackend.torch,
        )

    full_output, full_state = _run(q, k, v, xf, h0)

    first_output, first_state = _run(q[:, :split], k[:, :split], v[:, :split], xf[:, :split], h0)
    second_output, second_state = _run(q[:, split:], k[:, split:], v[:, split:], xf[:, split:], first_state)

    chunked_output = torch.cat([first_output, second_output], dim=1)

    assert_equal_tensors(chunked_output, full_output, False)
    assert_equal_tensors(second_state, full_state, False)


def _get_op_level_tensors(
    device: torch.device,
    dtype: torch.dtype,
    problem_shape: tuple[int, int, int, int, int, int, int],
    batch_size: int,
    sequence_length: int,
) -> tuple[dict, dict]:
    key_head_dim, value_head_dim, num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads = problem_shape

    q_kernel, q_torch = get_duplicated_tensors(
        (batch_size, sequence_length, num_q_heads, key_head_dim), device=device, dtype=dtype
    )
    k_kernel, k_torch = get_duplicated_tensors(
        (batch_size, sequence_length, num_k_heads, key_head_dim), device=device, dtype=dtype
    )
    v_kernel, v_torch = get_duplicated_tensors(
        (batch_size, sequence_length, num_v_heads, value_head_dim), device=device, dtype=dtype
    )
    xf_kernel, xf_torch = get_duplicated_tensors(
        (batch_size, sequence_length, num_f_heads), device=device, dtype=dtype
    )
    W_kernel, W_torch = get_duplicated_tensors(
        (num_weight_heads, value_head_dim, value_head_dim), device=device, dtype=dtype
    )

    kernel = dict(query=q_kernel, key=k_kernel, value=v_kernel, forget_input=xf_kernel, weight=W_kernel)
    torch_ref = dict(query=q_torch, key=k_torch, value=v_torch, forget_input=xf_torch, weight=W_torch)

    return kernel, torch_ref


@pytest.mark.parametrize("is_compiling", [False, True])
@pytest.mark.parametrize("has_input_state", [False, True])
@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_op_forward_kernel_vs_torch(
    dtype: torch.dtype,
    problem_shape: tuple[int, int, int, int, int, int, int],
    has_input_state: bool,
    is_compiling: bool,
) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    key_head_dim, value_head_dim, num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads = problem_shape
    num_heads = max(num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads)

    kernel, torch_ref = _get_op_level_tensors(device, dtype, problem_shape, batch_size=4, sequence_length=32)

    h0_kernel = h0_torch = None
    if has_input_state:
        h0_kernel, h0_torch = get_duplicated_tensors(
            (4, num_heads, key_head_dim, value_head_dim), device=device, dtype=dtype
        )

    m2rnn_fn = torch.compile(m2rnn, fullgraph=True) if is_compiling else m2rnn

    y_kernel, h_kernel = m2rnn_fn(**kernel, input_state=h0_kernel, kernel_backend=KernelBackend.triton)
    y_torch, h_torch = m2rnn(**torch_ref, input_state=h0_torch, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(y_kernel, y_torch, False)
    assert_equal_tensors(h_kernel, h_torch, False)


@pytest.mark.parametrize("is_compiling", [False, True])
@pytest.mark.parametrize("has_input_state", [False, True])
@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_op_backward_kernel_vs_torch(
    dtype: torch.dtype,
    problem_shape: tuple[int, int, int, int, int, int, int],
    has_input_state: bool,
    is_compiling: bool,
) -> None:
    """op-level equivalent of xma's `test_m2rnn` gradient checks: every input tensor's gradient
    (query, key, value, forget_input, weight, and the initial state when present) must match."""

    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    key_head_dim, value_head_dim, num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads = problem_shape
    num_heads = max(num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads)

    kernel, torch_ref = _get_op_level_tensors(device, dtype, problem_shape, batch_size=4, sequence_length=32)

    h0_kernel = h0_torch = None
    if has_input_state:
        h0_kernel, h0_torch = get_duplicated_tensors(
            (4, num_heads, key_head_dim, value_head_dim), device=device, dtype=dtype
        )

    m2rnn_fn = torch.compile(m2rnn, fullgraph=True) if is_compiling else m2rnn

    y_kernel, _ = m2rnn_fn(**kernel, input_state=h0_kernel, kernel_backend=KernelBackend.triton)
    y_torch, _ = m2rnn(**torch_ref, input_state=h0_torch, kernel_backend=KernelBackend.torch)

    y_kernel.sum().backward()
    y_torch.sum().backward()

    assert_equal_tensors(y_kernel, y_torch, False)
    for name in kernel:
        assert_equal_tensors(kernel[name].grad, torch_ref[name].grad, False)

    if has_input_state:
        assert_equal_tensors(h0_kernel.grad, h0_torch.grad, False)


@pytest.mark.parametrize("is_compiling", [False, True])
@pytest.mark.parametrize("has_input_state", [False, True])
@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_op_varlen_kernel_vs_torch(
    dtype: torch.dtype,
    problem_shape: tuple[int, int, int, int, int, int, int],
    has_input_state: bool,
    is_compiling: bool,
) -> None:
    """op-level equivalent of xma's `test_m2rnn` variable-length (packed, `cu_seqlens`) case: the
    torch reference has no varlen support, so it's reconstructed by running each sequence densely."""

    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    key_head_dim, value_head_dim, num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads = problem_shape
    num_heads = max(num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads)

    cu_seqlens = torch.tensor([0, 7, 19, 27, 93], device=device)
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    B = cu_seqlens.size(0) - 1
    total_tokens = int(cu_seqlens[-1])

    q_kernel, q_torch = get_duplicated_tensors((total_tokens, num_q_heads, key_head_dim), device=device, dtype=dtype)
    k_kernel, k_torch = get_duplicated_tensors((total_tokens, num_k_heads, key_head_dim), device=device, dtype=dtype)
    v_kernel, v_torch = get_duplicated_tensors((total_tokens, num_v_heads, value_head_dim), device=device, dtype=dtype)
    xf_kernel, xf_torch = get_duplicated_tensors((total_tokens, num_f_heads), device=device, dtype=dtype)
    W_kernel, W_torch = get_duplicated_tensors(
        (num_weight_heads, value_head_dim, value_head_dim), device=device, dtype=dtype
    )

    h0_kernel = h0_torch = None
    if has_input_state:
        h0_kernel, h0_torch = get_duplicated_tensors(
            (B, num_heads, key_head_dim, value_head_dim), device=device, dtype=dtype
        )

    m2rnn_fn = torch.compile(m2rnn, fullgraph=True) if is_compiling else m2rnn

    y_kernel, h_kernel = m2rnn_fn(
        query=q_kernel,
        key=k_kernel,
        value=v_kernel,
        forget_input=xf_kernel,
        weight=W_kernel,
        input_state=h0_kernel,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=KernelBackend.triton,
    )

    y_torch = []
    h_torch = []
    for i in range(B):
        y, h = m2rnn(
            query=q_torch[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
            key=k_torch[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
            value=v_torch[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
            forget_input=xf_torch[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
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
