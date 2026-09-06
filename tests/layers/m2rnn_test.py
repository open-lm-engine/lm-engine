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
_NUM_HEADS = 4
_KEY_HEAD_DIM = 8
_VALUE_HEAD_DIM = 8
_BATCH = 2
_PREFILL_LEN = 16


def _skip_unless_m2rnn_triton_available() -> torch.device:
    device = torch.device("cuda")
    skip_test_if_device_unavailable(device)

    if not is_triton_available():
        pytest.skip("skipping test because triton is unavailable")

    return device


def _make_m2rnn(device: torch.device) -> M2RNN:
    config = M2RNNArgs(
        k_head_dim=_KEY_HEAD_DIM,
        v_head_dim=_VALUE_HEAD_DIM,
        num_q_heads=_NUM_HEADS,
        num_k_heads=_NUM_HEADS,
        num_v_heads=_NUM_HEADS,
        num_f_heads=_NUM_HEADS,
        num_g_heads=_NUM_HEADS,
        num_weight_heads=_NUM_HEADS,
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
    m2rnn = M2RNN(
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
    ).to(device)
    m2rnn.eval()

    return m2rnn


def test_triton_forward_vs_torch_forward_prefill() -> None:
    device = _skip_unless_m2rnn_triton_available()
    m2rnn = _make_m2rnn(device)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)

    with enable_kernels([Kernel.m2rnn]):
        out_k = m2rnn(x)

    out_f = m2rnn(x)

    assert_close(out_k, out_f, rtol=1e-3, atol=1e-3)


def test_triton_forward_vs_torch_forward_decode() -> None:
    device = _skip_unless_m2rnn_triton_available()
    m2rnn = _make_m2rnn(device)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    x_gen = torch.randn(_BATCH, 1, _HIDDEN_SIZE, device=device)

    # seed the cache with a prefill, then compare a single incremental decoding step between the
    # two paths
    with enable_kernels([Kernel.m2rnn]):
        cache_k = GenerationCache()
        m2rnn(x, cache_params=cache_k)
        out_gen_k = m2rnn(x_gen, cache_params=cache_k)

    cache_f = GenerationCache()
    m2rnn(x, cache_params=cache_f)
    out_gen_f = m2rnn(x_gen, cache_params=cache_f)

    assert_close(out_gen_k, out_gen_f, rtol=1e-3, atol=1e-3)


def test_triton_forward_vs_torch_forward_backward() -> None:
    device = _skip_unless_m2rnn_triton_available()
    m2rnn = _make_m2rnn(device)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)

    # gradients w.r.t. the input should also match between the triton kernel and the torch
    # fallback; each path gets its own leaf input tensor so their .grad don't interfere
    x_k = x.clone().requires_grad_(True)
    with enable_kernels([Kernel.m2rnn]):
        out_k = m2rnn(x_k)
        out_k.sum().backward()

    x_f = x.clone().requires_grad_(True)
    out_f = m2rnn(x_f)
    out_f.sum().backward()

    assert_close(out_k, out_f, rtol=1e-3, atol=1e-3)
    assert x_k.grad is not None
    assert x_f.grad is not None
    assert_close(x_k.grad, x_f.grad, rtol=1e-3, atol=1e-3)


def test_m2rnn_torch_chunked_matches_full_sequence() -> None:
    torch.manual_seed(0)

    batch_size, sequence_length = 2, 10
    num_heads, key_head_dim, value_head_dim = 3, 4, 5
    split = 4

    q = torch.randn(batch_size, sequence_length, num_heads, key_head_dim)
    k = torch.randn(batch_size, sequence_length, num_heads, key_head_dim)
    v = torch.randn(batch_size, sequence_length, num_heads, value_head_dim)
    xf = torch.rand(batch_size, sequence_length, num_heads)
    W = torch.randn(num_heads, value_head_dim, value_head_dim)

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

    full_output, full_state = _run(q, k, v, xf, None)

    first_output, first_state = _run(q[:, :split], k[:, :split], v[:, :split], xf[:, :split], None)
    second_output, second_state = _run(q[:, split:], k[:, split:], v[:, split:], xf[:, split:], first_state)

    chunked_output = torch.cat([first_output, second_output], dim=1)

    assert_close(chunked_output, full_output, rtol=1e-5, atol=1e-5)
    assert_close(second_state, full_state, rtol=1e-5, atol=1e-5)
