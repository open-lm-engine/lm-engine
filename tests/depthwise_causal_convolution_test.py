# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch
from torch.testing import assert_close

from lm_engine.enums import Kernel
from lm_engine.kernels import enable_kernels
from lm_engine.modeling_utils import DepthwiseCausalConvolution
from lm_engine.utils import is_causal_conv1d_available

from .utils import skip_test_if_device_unavailable


_HIDDEN_SIZE = 8
_BATCH = 2
_PREFILL_LEN = 6


def _make_conv(
    hidden_size: int = _HIDDEN_SIZE, kernel_size: int = 4, activation: str | None = "silu", add_bias: bool = True
) -> DepthwiseCausalConvolution:
    return DepthwiseCausalConvolution(
        hidden_size=hidden_size,
        kernel_size=kernel_size,
        activation_function=activation,
        add_bias=add_bias,
        std=None,
        use_padding_free_transformer=False,
    )


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
@pytest.mark.parametrize("add_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
@pytest.mark.parametrize("output_state", [False, True])
@pytest.mark.parametrize("kernels", [[], [Kernel.causal_conv1d]])
@pytest.mark.parametrize("short_seq", [False, True])
def test_prefill_shapes(
    device: torch.device,
    kernel_size: int,
    add_bias: bool,
    activation: str | None,
    output_state: bool,
    kernels: list[Kernel],
    short_seq: bool,
) -> None:
    skip_test_if_device_unavailable(device)

    if Kernel.causal_conv1d in kernels and (device.type != "cuda" or not is_causal_conv1d_available()):
        pytest.skip("causal_conv1d unavailable")

    if Kernel.causal_conv1d in kernels and kernel_size == 1:
        pytest.skip("causal_conv1d only supports kernel_size between 2 and 4")

    with torch.device(device):
        conv = _make_conv(kernel_size=kernel_size, add_bias=add_bias, activation=activation)

    conv.eval()

    seq_len = max(1, kernel_size - 1) if short_seq else _PREFILL_LEN
    x = torch.randn(_BATCH, seq_len, _HIDDEN_SIZE, device=device)

    with enable_kernels(kernels):
        out, state = conv(x, input_state=None, attention_mask=None, output_state=output_state)

    assert out.size() == x.size()

    if output_state:
        assert state is not None
        assert state.size() == (_BATCH, _HIDDEN_SIZE, kernel_size)
    else:
        assert state is None


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
@pytest.mark.parametrize("add_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
@pytest.mark.parametrize("output_state", [False, True])
def test_generation_shapes(
    device: torch.device, kernel_size: int, add_bias: bool, activation: str | None, output_state: bool
) -> None:
    skip_test_if_device_unavailable(device)

    with torch.device(device):
        conv = _make_conv(kernel_size=kernel_size, add_bias=add_bias, activation=activation)

    conv.eval()

    x_prefill = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    _, state = conv(x_prefill, input_state=None, attention_mask=None, output_state=True)

    x_gen = torch.randn(_BATCH, 1, _HIDDEN_SIZE, device=device)
    out, state_out = conv(x_gen, input_state=state, attention_mask=None, output_state=output_state)

    assert out.size() == (_BATCH, 1, _HIDDEN_SIZE)

    if output_state:
        assert state_out is not None
        assert state_out.size() == state.size()
    else:
        assert state_out is None


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
@pytest.mark.parametrize("add_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
@pytest.mark.parametrize("seq_len", [1, 2, 4])
def test_zero_state_matches_fresh_prefill(
    device: torch.device,
    kernel_size: int,
    add_bias: bool,
    activation: str | None,
    seq_len: int,
) -> None:
    skip_test_if_device_unavailable(device)

    with torch.device(device):
        conv = _make_conv(kernel_size=kernel_size, add_bias=add_bias, activation=activation)

    conv.eval()

    x = torch.randn(_BATCH, seq_len, _HIDDEN_SIZE, device=device)
    zero_state = torch.zeros(_BATCH, _HIDDEN_SIZE, kernel_size, device=device)

    out_fresh, state_fresh = conv(x, input_state=None, attention_mask=None, output_state=True)
    out_zero, state_zero = conv(x, input_state=zero_state, attention_mask=None, output_state=True)

    assert_close(out_zero, out_fresh, rtol=1e-5, atol=1e-5)
    assert_close(state_zero, state_fresh, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
@pytest.mark.parametrize("add_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
@pytest.mark.parametrize("continuation_len", [1, 2, 4])
@pytest.mark.parametrize("n_gen_steps", [1, 2])
@pytest.mark.parametrize("short_prefill", [False, True])
def test_consistency(
    device: torch.device,
    kernel_size: int,
    add_bias: bool,
    activation: str | None,
    continuation_len: int,
    n_gen_steps: int,
    short_prefill: bool,
) -> None:
    """prefill → N generation steps must match the same positions in a full-sequence forward pass."""
    skip_test_if_device_unavailable(device)

    with torch.device(device):
        conv = _make_conv(kernel_size=kernel_size, add_bias=add_bias, activation=activation)

    conv.eval()

    prefill_len = max(1, kernel_size - 1) if short_prefill else _PREFILL_LEN
    total_gen_steps = continuation_len + n_gen_steps
    x_full = torch.randn(_BATCH, prefill_len + total_gen_steps, _HIDDEN_SIZE, device=device)

    out_full, state_full = conv(x_full, input_state=None, attention_mask=None, output_state=True)

    _, state = conv(x_full[:, :prefill_len], input_state=None, attention_mask=None, output_state=True)
    out_continue, state = conv(
        x_full[:, prefill_len : prefill_len + continuation_len],
        input_state=state,
        attention_mask=None,
        output_state=True,
    )
    assert_close(
        out_continue,
        out_full[:, prefill_len : prefill_len + continuation_len],
        rtol=1e-5,
        atol=1e-5,
    )

    for step in range(n_gen_steps):
        start = prefill_len + continuation_len + step
        x_step = x_full[:, start : start + 1]
        is_last = step == n_gen_steps - 1
        out_step, state = conv(x_step, input_state=state, attention_mask=None, output_state=not is_last)
        assert_close(out_step, out_full[:, start : start + 1], rtol=1e-5, atol=1e-5)

    assert state is None
    _, state = conv(x_full[:, :prefill_len], input_state=None, attention_mask=None, output_state=True)
    _, state = conv(
        x_full[:, prefill_len:],
        input_state=state,
        attention_mask=None,
        output_state=True,
    )
    assert_close(state, state_full, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
def test_attention_mask(device: torch.device, kernel_size: int) -> None:
    skip_test_if_device_unavailable(device)
    conv = _make_conv(kernel_size=kernel_size, activation=None).to(device)
    conv.eval()

    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)

    # all-ones mask is a no-op
    mask_ones = torch.ones(_BATCH, _PREFILL_LEN, dtype=x.dtype, device=device)
    out_no_mask, _ = conv(x, input_state=None, attention_mask=None, output_state=False)
    out_ones, _ = conv(x, input_state=None, attention_mask=mask_ones, output_state=False)
    assert_close(out_no_mask, out_ones, rtol=1e-5, atol=1e-5)

    # padding positions in the output must be exactly zero
    mask = mask_ones.clone()
    mask[1, :3] = 0
    out_masked, _ = conv(x, input_state=None, attention_mask=mask, output_state=False)
    assert (out_masked[1, :3] == 0).all()

    # non-padding positions must match a manually zeroed input (no mask)
    x_zeroed = x.clone()
    x_zeroed[1, :3] = 0
    out_zeroed, _ = conv(x_zeroed, input_state=None, attention_mask=None, output_state=False)
    assert_close(out_masked[0], out_zeroed[0], rtol=1e-5, atol=1e-5)
    assert_close(out_masked[1, 3:], out_zeroed[1, 3:], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
def test_kernel_vs_fallback(device: torch.device, kernel_size: int, activation: str | None) -> None:
    skip_test_if_device_unavailable(device)

    if device.type != "cuda" or not is_causal_conv1d_available():
        pytest.skip("causal_conv1d unavailable")

    if kernel_size == 1:
        pytest.skip("causal_conv1d only supports kernel_size between 2 and 4")

    conv = _make_conv(kernel_size=kernel_size, activation=activation).to(device)
    conv.eval()

    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    x_gen = torch.randn(_BATCH, 1, _HIDDEN_SIZE, device=device)

    with enable_kernels([Kernel.causal_conv1d]):
        out_k, state_k = conv(x, input_state=None, attention_mask=None, output_state=True)
        out_gen_k, _ = conv(x_gen, input_state=state_k, attention_mask=None, output_state=False)

    out_f, state_f = conv(x, input_state=None, attention_mask=None, output_state=True)
    out_gen_f, _ = conv(x_gen, input_state=state_f, attention_mask=None, output_state=False)

    assert_close(out_k, out_f, rtol=1e-5, atol=1e-5)
    assert_close(state_k, state_f, rtol=1e-5, atol=1e-5)
    assert_close(out_gen_k, out_gen_f, rtol=1e-5, atol=1e-5)
