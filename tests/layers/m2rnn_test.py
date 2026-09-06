# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch
from torch.testing import assert_close

from lm_engine.accelerator import KernelBackend
from lm_engine.enums import Kernel
from lm_engine.generation_cache import GenerationCache
from lm_engine.kernels import enable_kernels
from lm_engine.modeling_utils.sequence_mixer_blocks import M2RNN, M2RNNArgs, m2rnn
from lm_engine.utils import is_triton_available
from tests.utils import skip_test_if_device_unavailable


_HIDDEN_SIZE = 32
_BATCH = 2
_PREFILL_LEN = 16
_DTYPES = [torch.float32, torch.bfloat16]
_TOLERANCES = {torch.float32: dict(rtol=1e-5, atol=1e-5), torch.bfloat16: dict(rtol=1e-2, atol=1e-2)}


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


def _skip_unless_m2rnn_triton_available() -> torch.device:
    device = torch.device("cuda")
    skip_test_if_device_unavailable(device)

    if not is_triton_available():
        pytest.skip("skipping test because triton is unavailable")

    return device


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
    device = _skip_unless_m2rnn_triton_available()
    m2rnn_module = _make_m2rnn(device, dtype, problem_shape)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device, dtype=dtype)

    with enable_kernels([Kernel.m2rnn]):
        out_k = m2rnn_module(x)

    out_f = m2rnn_module(x)

    assert_close(out_k, out_f, **_TOLERANCES[dtype])


@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_decode_vs_torch_decode(
    dtype: torch.dtype, problem_shape: tuple[int, int, int, int, int, int, int]
) -> None:
    device = _skip_unless_m2rnn_triton_available()
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

    assert_close(out_gen_k, out_gen_f, **_TOLERANCES[dtype])


@pytest.mark.parametrize("problem_shape", _PROBLEM_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_vs_torch_forward_backward(
    dtype: torch.dtype, problem_shape: tuple[int, int, int, int, int, int, int]
) -> None:
    device = _skip_unless_m2rnn_triton_available()
    m2rnn_module = _make_m2rnn(device, dtype, problem_shape)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device, dtype=dtype)

    x_k = x.clone().requires_grad_(True)
    with enable_kernels([Kernel.m2rnn]):
        out_k = m2rnn_module(x_k)
        out_k.sum().backward()

    x_f = x.clone().requires_grad_(True)
    out_f = m2rnn_module(x_f)
    out_f.sum().backward()

    tolerances = _TOLERANCES[dtype]
    assert_close(out_k, out_f, **tolerances)
    assert x_k.grad is not None
    assert x_f.grad is not None
    assert_close(x_k.grad, x_f.grad, **tolerances)


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

    tolerances = _TOLERANCES[dtype]
    assert_close(chunked_output, full_output, **tolerances)
    assert_close(second_state, full_state, **tolerances)
