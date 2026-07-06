# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch
from torch.testing import assert_close

from lm_engine.enums import Kernel
from lm_engine.generation_cache import GenerationCache
from lm_engine.kernels import enable_kernels
from lm_engine.modeling_utils.sequence_mixer_blocks.mamba2 import Mamba2, Mamba2Args
from lm_engine.utils import is_mamba_2_ssm_available

from .utils import skip_test_if_device_unavailable


_HIDDEN_SIZE = 64
_INTERMEDIATE_SIZE = 128
_NUM_HEADS = 8
_STATE_SIZE = 16
_NUM_GROUPS = 1
_CHUNK_SIZE = 16
_BATCH = 2
_PREFILL_LEN = 32  # must be a multiple of chunk_size


def _skip_unless_mamba2_ssm_available() -> torch.device:
    device = torch.device("cuda")
    skip_test_if_device_unavailable(device)

    if not is_mamba_2_ssm_available():
        pytest.skip("skipping test because mamba_ssm is unavailable")

    return device


def _make_mamba2(device: torch.device) -> Mamba2:
    config = Mamba2Args(
        state_size=_STATE_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        num_heads=_NUM_HEADS,
        conv_kernel_size=4,
        activation_function="silu",
        num_groups=_NUM_GROUPS,
        chunk_size=_CHUNK_SIZE,
        normalization_function="rmsnorm",
    )

    torch.manual_seed(42)
    mamba2 = Mamba2(
        hidden_size=_HIDDEN_SIZE,
        config=config,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        m_width=1.0,
        init_method="normal",
        num_layers=1,
        layer_idx=0,
        use_depth_scaled_init=False,
    ).to(device)
    mamba2.eval()

    return mamba2


def test_cuda_forward_vs_torch_forward_prefill() -> None:
    device = _skip_unless_mamba2_ssm_available()
    mamba2 = _make_mamba2(device)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)

    # a full-sequence forward (no cache) through the fused CUDA kernels (_cuda_forward) and
    # through the naive torch fallback (_torch_forward) should compute the same function
    with enable_kernels([Kernel.mamba2_ssm]):
        out_k = mamba2(x)

    out_f = mamba2(x)

    assert_close(out_k, out_f, rtol=1e-3, atol=1e-3)


def test_cuda_forward_vs_torch_forward_decode() -> None:
    device = _skip_unless_mamba2_ssm_available()
    mamba2 = _make_mamba2(device)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    x_gen = torch.randn(_BATCH, 1, _HIDDEN_SIZE, device=device)

    # seed the cache with a prefill, then compare a single incremental decoding step (the
    # cached/precomputed-state branch of each implementation) between the two paths
    with enable_kernels([Kernel.mamba2_ssm]):
        cache_k = GenerationCache()
        mamba2(x, cache_params=cache_k)
        out_gen_k = mamba2(x_gen, cache_params=cache_k)

    cache_f = GenerationCache()
    mamba2(x, cache_params=cache_f)
    out_gen_f = mamba2(x_gen, cache_params=cache_f)

    assert_close(out_gen_k, out_gen_f, rtol=1e-3, atol=1e-3)


def test_cuda_forward_vs_torch_forward_backward() -> None:
    device = _skip_unless_mamba2_ssm_available()
    mamba2 = _make_mamba2(device)

    torch.manual_seed(0)
    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)

    # gradients w.r.t. the input should also match between the fused kernel and the torch
    # fallback; each path gets its own leaf input tensor so their .grad don't interfere
    x_k = x.clone().requires_grad_(True)
    with enable_kernels([Kernel.mamba2_ssm]):
        out_k = mamba2(x_k)
        out_k.sum().backward()

    x_f = x.clone().requires_grad_(True)
    out_f = mamba2(x_f)
    out_f.sum().backward()

    assert_close(out_k, out_f, rtol=1e-3, atol=1e-3)
    assert x_k.grad is not None
    assert x_f.grad is not None
    assert_close(x_k.grad, x_f.grad, rtol=1e-3, atol=1e-3)
