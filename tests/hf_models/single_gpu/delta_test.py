# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import pytest
import torch
from einops import repeat

from lm_engine.utils import is_fla_available


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_fla_available(),
    reason="CUDA and FLA are required",
)

if is_fla_available():
    from lm_engine.hf_models.modeling_utils.mlp_blocks.delta_utils import chunk_delta_rule


def _leaf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def prepare_data(
    B: int,
    T: int,
    H: int,
    DK: int,
    DV: int,
    HK: int,
    HV: int,
    HB: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(B, T, HK, DK, dtype=dtype, device=device)
    k = torch.randn(B, T, HK, DK, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, DV, dtype=dtype, device=device)
    b = torch.randn(B, T, HB, dtype=dtype, device=device).sigmoid()
    S = torch.randn(1, H, DK, DV, dtype=dtype, device=device)
    return q, k, v, b, S


@pytest.mark.parametrize(
    "HK, HV, HB",
    [
        (1, 4, 8),
        (4, 1, 8),
        (8, 4, 1),
        (2, 8, 4),
        (4, 8, 2),
        (8, 2, 4),
        (1, 1, 7),
        (7, 1, 1),
        (1, 7, 1),
        (5, 5, 1),
        (1, 5, 5),
        (5, 1, 5),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
)
def test_broadcast(HK: int, HV: int, HB: int, dtype: torch.dtype) -> None:
    B, T, DK, DV = 7, 1024, 128, 64
    H = max(HK, HV, HB)
    assert H % HK == 0
    assert H % HV == 0
    assert H % HB == 0
    q, k, v, b, S = prepare_data(
        B=B,
        T=T,
        H=H,
        DK=DK,
        DV=DV,
        HK=HK,
        HV=HV,
        HB=HB,
        dtype=dtype,
        device="cuda",
    )
    q0 = _leaf(q)
    k0 = _leaf(k)
    v0 = _leaf(v)
    b0 = _leaf(b)
    S0 = _leaf(S)
    o0, S0_ = chunk_delta_rule(
        q=q0,
        k=k0,
        v=v0,
        beta=b0,
        initial_state=S0,
        output_final_state=False,
        cu_seqlens=None,
        use_q_l2norm_in_kernel=True,
        use_k_l2norm_in_kernel=True,
        allow_fp32=True,
    )

    q1 = _leaf(q)
    k1 = _leaf(k)
    v1 = _leaf(v)
    b1 = _leaf(b)
    S1 = _leaf(S)

    q1_broadcast = repeat(q1, "... h d -> ... (h g) d", g=H // HK)
    k1_broadcast = repeat(k1, "... h d -> ... (h g) d", g=H // HK)
    v1_broadcast = repeat(v1, "... h d -> ... (h g) d", g=H // HV)
    b1_broadcast = repeat(b1, "... h   -> ... (h g)  ", g=H // HB)
    S1_broadcast = repeat(S1, "1 ... -> b ...", b=B)

    o1, S1_ = chunk_delta_rule(
        q=q1_broadcast,
        k=k1_broadcast,
        v=v1_broadcast,
        beta=b1_broadcast,
        initial_state=S1_broadcast,
        output_final_state=False,
        cu_seqlens=None,
        use_q_l2norm_in_kernel=True,
        use_k_l2norm_in_kernel=True,
        allow_fp32=True,
    )

    (o0 / max(DK, DV)).sum().backward()
    (o1 / max(DK, DV)).sum().backward()

    torch.testing.assert_close(o0, o1)
    torch.testing.assert_close(S0_, S1_)
    torch.testing.assert_close(q0.grad, q1.grad)
    torch.testing.assert_close(k0.grad, k1.grad)
    torch.testing.assert_close(v0.grad, v1.grad)
    torch.testing.assert_close(b0.grad, b1.grad)
    torch.testing.assert_close(S0.grad, S1.grad)
