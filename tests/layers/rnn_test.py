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
from lm_engine.modeling_utils.sequence_mixer_blocks import RNN, RNNArgs, rnn
from lm_engine.utils import is_triton_available
from tests.utils import skip_test_if_device_unavailable


_HIDDEN_SIZE = 32
_NUM_HEADS = 4
_STATE_HEAD_DIM = 8
_BATCH = 2
_PREFILL_LEN = 16


def _skip_unless_rnn_triton_available() -> torch.device:
    device = torch.device("cuda")
    skip_test_if_device_unavailable(device)

    if not is_triton_available():
        pytest.skip("skipping test because triton is unavailable")

    return device


def _make_rnn(device: torch.device) -> RNN:
    config = RNNArgs(
        state_head_dim=_STATE_HEAD_DIM,
        num_input_heads=_NUM_HEADS,
        num_weight_heads=_NUM_HEADS,
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
    ).to(device)
    rnn_module.eval()

    return rnn_module


def test_triton_prefill_vs_torch_prefill() -> None:
    device = _skip_unless_rnn_triton_available()
    rnn_module = _make_rnn(device)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)

    with enable_kernels([Kernel.rnn]):
        out_k = rnn_module(x)

    out_f = rnn_module(x)

    assert_close(out_k, out_f, rtol=1e-3, atol=1e-3)


def test_triton_decode_vs_torch_decode() -> None:
    device = _skip_unless_rnn_triton_available()
    rnn_module = _make_rnn(device)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    x_gen = torch.randn(_BATCH, 1, _HIDDEN_SIZE, device=device)

    with enable_kernels([Kernel.rnn]):
        cache_k = GenerationCache()
        rnn_module(x, cache_params=cache_k)
        out_gen_k = rnn_module(x_gen, cache_params=cache_k)

    cache_f = GenerationCache()
    rnn_module(x, cache_params=cache_f)
    out_gen_f = rnn_module(x_gen, cache_params=cache_f)

    assert_close(out_gen_k, out_gen_f, rtol=1e-3, atol=1e-3)


def test_triton_vs_torch_forward_backward() -> None:
    device = _skip_unless_rnn_triton_available()
    rnn_module = _make_rnn(device)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)

    # gradients w.r.t. the input should also match between the triton kernel and the torch
    # fallback; each path gets its own leaf input tensor so their .grad don't interfere
    x_k = x.clone().requires_grad_(True)
    with enable_kernels([Kernel.rnn]):
        out_k = rnn_module(x_k)
        out_k.sum().backward()

    x_f = x.clone().requires_grad_(True)
    out_f = rnn_module(x_f)
    out_f.sum().backward()

    assert_close(out_k, out_f, rtol=1e-3, atol=1e-3)
    assert x_k.grad is not None
    assert x_f.grad is not None
    assert_close(x_k.grad, x_f.grad, rtol=1e-3, atol=1e-3)


def test_torch_prefill_continuation() -> None:
    torch.manual_seed(0)

    batch_size, sequence_length = 2, 10
    num_heads, state_head_dim = 3, 4
    split = 4

    x = torch.randn(batch_size, sequence_length, num_heads, state_head_dim)
    W = torch.randn(num_heads, state_head_dim, state_head_dim)

    def _run(x, h0):
        return rnn(input=x, weight=W, input_state=h0, kernel_backend=KernelBackend.torch)

    full_output, full_state = _run(x, None)

    first_output, first_state = _run(x[:, :split], None)
    second_output, second_state = _run(x[:, split:], first_state)

    chunked_output = torch.cat([first_output, second_output], dim=1)

    assert_close(chunked_output, full_output, rtol=1e-5, atol=1e-5)
    assert_close(second_state, full_state, rtol=1e-5, atol=1e-5)
