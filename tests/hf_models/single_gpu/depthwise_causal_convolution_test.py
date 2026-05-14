# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import pytest
import torch
from torch.testing import assert_close

from lm_engine.enums import Kernel
from lm_engine.hf_models.modeling_utils.depthwise_causal_convolution import DepthwiseCausalConvolution
from lm_engine.kernels import enable_kernels
from lm_engine.utils import is_causal_conv1d_available

from ...utils import skip_test_if_device_unavailable


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
def test_prefill_shapes(
    device: torch.device,
    kernel_size: int,
    add_bias: bool,
    activation: str | None,
    output_state: bool,
    kernels: list[Kernel],
) -> None:
    skip_test_if_device_unavailable(device)

    if Kernel.causal_conv1d in kernels and not is_causal_conv1d_available():
        pytest.skip("skipping test because causal_conv1d is unavailable")

    with torch.device(device):
        conv = _make_conv(kernel_size=kernel_size, add_bias=add_bias, activation=activation)

    conv.eval()

    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)

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
def test_prefill_short_sequence_state(device: torch.device, kernel_size: int) -> None:
    """Prefill with seq_len < kernel_size still produces correct state shape (zero-padded)."""
    skip_test_if_device_unavailable(device)

    with torch.device(device):
        conv = _make_conv(kernel_size=kernel_size)

    conv.eval()

    short_len = max(1, kernel_size - 1)
    x = torch.randn(_BATCH, short_len, _HIDDEN_SIZE, device=device)
    _, state = conv(x, input_state=None, attention_mask=None, output_state=True)

    assert state is not None
    assert state.size() == (_BATCH, _HIDDEN_SIZE, kernel_size)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
@pytest.mark.parametrize("add_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
def test_generation_output_shape(
    device: torch.device, kernel_size: int, add_bias: bool, activation: str | None
) -> None:
    skip_test_if_device_unavailable(device)
    conv = _make_conv(kernel_size=kernel_size, add_bias=add_bias, activation=activation).to(device)
    conv.eval()

    x_prefill = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    _, state = conv(x_prefill, input_state=None, attention_mask=None, output_state=True)

    x_gen = torch.randn(_BATCH, 1, _HIDDEN_SIZE, device=device)
    out, state_out = conv(x_gen, input_state=state, attention_mask=None, output_state=False)

    assert out.shape == (_BATCH, 1, _HIDDEN_SIZE)
    assert state_out is None


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
def test_generation_output_state_true(device: torch.device, kernel_size: int) -> None:
    """output_state=True during generation returns an updated state of the same shape."""
    skip_test_if_device_unavailable(device)
    conv = _make_conv(kernel_size=kernel_size).to(device)
    conv.eval()

    x_prefill = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    _, state = conv(x_prefill, input_state=None, attention_mask=None, output_state=True)

    x_gen = torch.randn(_BATCH, 1, _HIDDEN_SIZE, device=device)
    out, state_out = conv(x_gen, input_state=state, attention_mask=None, output_state=True)

    assert out.shape == (_BATCH, 1, _HIDDEN_SIZE)
    assert state_out is not None
    assert state_out.shape == state.shape


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
@pytest.mark.parametrize("add_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
def test_prefill_then_generation_matches_full_sequence(
    device: torch.device, kernel_size: int, add_bias: bool, activation: str | None
) -> None:
    """prefill(L) → generation(1) must equal position L of prefill(L+1)."""
    skip_test_if_device_unavailable(device)
    conv = _make_conv(kernel_size=kernel_size, add_bias=add_bias, activation=activation).to(device)
    conv.eval()

    x_full = torch.randn(_BATCH, _PREFILL_LEN + 1, _HIDDEN_SIZE, device=device)
    x_prefill = x_full[:, :_PREFILL_LEN]
    x_gen = x_full[:, _PREFILL_LEN:]

    out_full, _ = conv(x_full, input_state=None, attention_mask=None, output_state=False)
    expected = out_full[:, -1:]

    _, state = conv(x_prefill, input_state=None, attention_mask=None, output_state=True)
    out_gen, _ = conv(x_gen, input_state=state, attention_mask=None, output_state=False)

    assert_close(out_gen, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
def test_multi_step_generation_matches_full_sequence(device: torch.device, kernel_size: int) -> None:
    """Two sequential generation steps must agree with the full-sequence prefill."""
    skip_test_if_device_unavailable(device)
    conv = _make_conv(kernel_size=kernel_size).to(device)
    conv.eval()

    x_full = torch.randn(_BATCH, _PREFILL_LEN + 2, _HIDDEN_SIZE, device=device)
    x_prefill = x_full[:, :_PREFILL_LEN]
    x_gen1 = x_full[:, _PREFILL_LEN : _PREFILL_LEN + 1]
    x_gen2 = x_full[:, _PREFILL_LEN + 1 :]

    out_full, _ = conv(x_full, input_state=None, attention_mask=None, output_state=False)

    _, state = conv(x_prefill, input_state=None, attention_mask=None, output_state=True)
    out1, state = conv(x_gen1, input_state=state, attention_mask=None, output_state=True)
    out2, _ = conv(x_gen2, input_state=state, attention_mask=None, output_state=False)

    assert_close(out1, out_full[:, _PREFILL_LEN : _PREFILL_LEN + 1], rtol=1e-5, atol=1e-5)
    assert_close(out2, out_full[:, _PREFILL_LEN + 1 :], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
def test_short_prefill_then_generation_matches_full_sequence(device: torch.device, kernel_size: int) -> None:
    """seq_len < kernel_size at prefill: generation still matches full-sequence output."""
    skip_test_if_device_unavailable(device)
    conv = _make_conv(kernel_size=kernel_size).to(device)
    conv.eval()

    short_len = max(1, kernel_size - 1)
    x_full = torch.randn(_BATCH, short_len + 1, _HIDDEN_SIZE, device=device)
    x_prefill = x_full[:, :short_len]
    x_gen = x_full[:, short_len:]

    out_full, _ = conv(x_full, input_state=None, attention_mask=None, output_state=False)
    expected = out_full[:, -1:]

    _, state = conv(x_prefill, input_state=None, attention_mask=None, output_state=True)
    out_gen, _ = conv(x_gen, input_state=state, attention_mask=None, output_state=False)

    assert_close(out_gen, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
def test_attention_mask_does_not_affect_non_padding_items(device: torch.device, kernel_size: int) -> None:
    """A batch item with no padding should produce the same output regardless of mask."""
    skip_test_if_device_unavailable(device)
    conv = _make_conv(kernel_size=kernel_size).to(device)
    conv.eval()

    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    mask_all_ones = torch.ones(_BATCH, _PREFILL_LEN, dtype=x.dtype, device=device)

    out_no_mask, _ = conv(x, input_state=None, attention_mask=None, output_state=False)
    out_ones_mask, _ = conv(x, input_state=None, attention_mask=mask_all_ones, output_state=False)

    assert_close(out_no_mask, out_ones_mask, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
def test_attention_mask_zeroes_padding_output(device: torch.device, kernel_size: int) -> None:
    """Output at masked (padding) positions must be exactly zero."""
    skip_test_if_device_unavailable(device)
    conv = _make_conv(kernel_size=kernel_size, activation=None).to(device)
    conv.eval()

    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    mask = torch.ones(_BATCH, _PREFILL_LEN, dtype=x.dtype, device=device)
    mask[1, :3] = 0  # batch item 1: first 3 tokens are padding

    out_masked, _ = conv(x, input_state=None, attention_mask=mask, output_state=False)

    # Padding positions in the output must be zeroed
    assert (out_masked[1, :3] == 0).all()


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
def test_attention_mask_non_padding_matches_zeroed_input(device: torch.device, kernel_size: int) -> None:
    """Non-padding output positions must equal the output from a manually zeroed input (no mask)."""
    skip_test_if_device_unavailable(device)
    conv = _make_conv(kernel_size=kernel_size, activation=None).to(device)
    conv.eval()

    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    mask = torch.ones(_BATCH, _PREFILL_LEN, dtype=x.dtype, device=device)
    mask[1, :3] = 0  # batch item 1: first 3 tokens are padding

    x_zeroed = x.clone()
    x_zeroed[1, :3] = 0

    out_masked, _ = conv(x, input_state=None, attention_mask=mask, output_state=False)
    out_zeroed, _ = conv(x_zeroed, input_state=None, attention_mask=None, output_state=False)

    # Non-padding positions and the unaffected batch item must agree
    assert_close(out_masked[0], out_zeroed[0], rtol=1e-5, atol=1e-5)
    assert_close(out_masked[1, 3:], out_zeroed[1, 3:], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not is_causal_conv1d_available(), reason="causal_conv1d not installed")
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
def test_kernel_path_matches_fallback_prefill(device: torch.device, kernel_size: int, activation: str | None) -> None:
    skip_test_if_device_unavailable(device)
    conv = _make_conv(kernel_size=kernel_size, activation=activation).to(device)
    conv.eval()

    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)

    with enable_kernels([Kernel.causal_conv1d]):
        out_kernel, state_kernel = conv(x, input_state=None, attention_mask=None, output_state=True)

    out_fallback, state_fallback = conv(x, input_state=None, attention_mask=None, output_state=True)

    assert_close(out_kernel, out_fallback, rtol=1e-5, atol=1e-5)
    assert_close(state_kernel, state_fallback, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not is_causal_conv1d_available(), reason="causal_conv1d not installed")
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
@pytest.mark.parametrize("kernel_size", [1, 4])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
def test_kernel_path_matches_fallback_generation(
    device: torch.device, kernel_size: int, activation: str | None
) -> None:
    skip_test_if_device_unavailable(device)
    conv = _make_conv(kernel_size=kernel_size, activation=activation).to(device)
    conv.eval()

    x_prefill = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    x_gen = torch.randn(_BATCH, 1, _HIDDEN_SIZE, device=device)

    with enable_kernels([Kernel.causal_conv1d]):
        _, state_k = conv(x_prefill, input_state=None, attention_mask=None, output_state=True)
        out_kernel, _ = conv(x_gen, input_state=state_k, attention_mask=None, output_state=False)

    _, state_f = conv(x_prefill, input_state=None, attention_mask=None, output_state=True)
    out_fallback, _ = conv(x_gen, input_state=state_f, attention_mask=None, output_state=False)

    assert_close(out_kernel, out_fallback, rtol=1e-5, atol=1e-5)
