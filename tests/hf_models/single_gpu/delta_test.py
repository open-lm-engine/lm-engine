# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch
from einops import rearrange, repeat
from quack.rmsnorm import rmsnorm

import lm_engine.hf_models.modeling_utils.mlp_blocks.delta_mlp as delta_mlp_module
from lm_engine.hf_models.modeling_utils.mlp_blocks.delta_mlp import DeltaMLP
from lm_engine.hf_models.modeling_utils.normalization import get_normalization_function
from lm_engine.utils import is_fla_available


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_fla_available(),
    reason="CUDA and FLA are required",
)

if is_fla_available():
    from lm_engine.hf_models.modeling_utils.mlp_blocks.delta_mlp.utils import chunk_delta_rule


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


def _delta_mlp_kwargs(**overrides) -> dict:
    kwargs = dict(
        hidden_size=32,
        intermediate_size=64,
        activation_function="silu",
        add_bias=False,
        dropout=0,
        use_interleaved_weights=False,
        num_ranks=8,
        num_heads=4,
        use_v_proj=False,
        use_q_l2norm=True,
        use_shortconv=False,
        use_tied_beta=False,
        use_decay_beta=False,
        use_head_o_norm=False,
        allow_neg_eigval=False,
        conv_size=4,
        layer_idx=0,
        norm_eps=1e-6,
        init_method="normal",
        initializer_range=0.02,
        m_width=None,
        A_init_min=0,
        A_init_max=16,
        dt_init_min=0.001,
        dt_init_max=0.1,
        dt_init_floor=1e-4,
        num_layers=1,
        use_depth_scaled_init=False,
        value_scale=None,
        use_v_silu=True,
        use_v_norm=False,
        sequence_parallel=False,
    )
    kwargs.update(overrides)
    return kwargs


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
    "use_v_norm",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "use_o_norm",
    [
        True,
        False,
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
def test_broadcast(HK: int, HV: int, HB: int, use_v_norm: bool, use_o_norm: bool, dtype: torch.dtype) -> None:
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

    if use_v_norm:
        v_norm0 = get_normalization_function("rmsnorm", H * DV, eps=1e-6).to(dtype=dtype, device="cuda")
        v_norm1 = get_normalization_function("rmsnorm", H * DV, eps=1e-6).to(dtype=dtype, device="cuda")
        v_norm_w = torch.randn(H * DV, dtype=dtype, device="cuda")
        v_norm_w0 = _leaf(v_norm_w)
        v_norm_w1 = _leaf(v_norm_w)
        with torch.no_grad():
            v_norm0.weight.copy_(v_norm_w0)
            v_norm1.weight.copy_(v_norm_w1)
    else:
        v_norm0 = None
        v_norm1 = None

    if use_o_norm:
        o_norm0 = get_normalization_function("rmsnorm", H * DV, eps=1e-6).to(dtype=dtype, device="cuda")
        o_norm1 = get_normalization_function("rmsnorm", H * DV, eps=1e-6).to(dtype=dtype, device="cuda")
        o_norm_w = torch.randn(H * DV, dtype=dtype, device="cuda")
        o_norm_w0 = _leaf(o_norm_w)
        o_norm_w1 = _leaf(o_norm_w)
        with torch.no_grad():
            o_norm0.weight.copy_(o_norm_w0)
            o_norm1.weight.copy_(o_norm_w1)
    else:
        o_norm0 = None
        o_norm1 = None

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
        v_norm=v_norm0,
        o_norm=o_norm0,
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

    if use_v_norm:
        assert v_norm1 is not None
        v1_broadcast = rearrange(v1_broadcast, "b t h d -> (b t) (h d)", b=B, t=T, h=H, d=DV)
        # v1_broadcast = v_norm1(v1_broadcast)
        v1_broadcast = rmsnorm(v1_broadcast, weight=v_norm1.weight, eps=v_norm1.eps)
        v1_broadcast = rearrange(v1_broadcast, "(b t) (h d) -> b t h d", b=B, t=T, h=H, d=DV)

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
        v_norm=None,
        o_norm=None,
        allow_fp32=True,
    )

    if use_o_norm:
        assert o_norm1 is not None
        o1 = rearrange(o1, "b t h d -> (b t) (h d)", b=B, t=T, h=H, d=DV)
        # o1 = o_norm1(o1)
        o1 = rmsnorm(o1, weight=o_norm1.weight, eps=o_norm1.eps)
        o1 = rearrange(o1, "(b t) (h d) -> b t h d", b=B, t=T, h=H, d=DV)

    (o0 / max(DK, DV)).sum().backward()
    (o1 / max(DK, DV)).sum().backward()

    torch.testing.assert_close(o0, o1)
    torch.testing.assert_close(S0_, S1_)
    torch.testing.assert_close(q0.grad, q1.grad)
    torch.testing.assert_close(k0.grad, k1.grad)
    torch.testing.assert_close(v0.grad, v1.grad)
    torch.testing.assert_close(b0.grad, b1.grad)
    torch.testing.assert_close(S0.grad, S1.grad)
    if use_v_norm:
        torch.testing.assert_close(v_norm0.weight.grad, v_norm1.weight.grad)
    if use_o_norm:
        torch.testing.assert_close(o_norm0.weight.grad, o_norm1.weight.grad)


@pytest.mark.parametrize("use_shortconv", [False, True])
@pytest.mark.parametrize(
    ("lengths", "padded_seqlen"),
    [
        pytest.param([97, 65], 97, id="multi_chunk"),
        pytest.param([33, 17], 65, id="short_valid_chunk"),
        pytest.param([97], 97, id="single_chunk"),
    ],
)
def test_packed_input_equivalence(
    use_shortconv: bool,
    lengths: list[int],
    padded_seqlen: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1234)

    dtype = torch.bfloat16
    device = torch.device("cuda")
    model_kwargs = _delta_mlp_kwargs(use_shortconv=use_shortconv)
    hidden_size = model_kwargs["hidden_size"]
    max_seqlen = max(lengths)
    assert padded_seqlen >= max_seqlen

    inputs = [torch.randn(length, hidden_size, dtype=dtype, device=device) for length in lengths]
    padded_input = torch.zeros(
        len(lengths),
        padded_seqlen,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    attention_mask = torch.zeros(
        len(lengths),
        padded_seqlen,
        dtype=torch.int,
        device=device,
    )
    for i, x in enumerate(inputs):
        pad_tokens = padded_seqlen - x.size(0)
        padded_input[i, pad_tokens:] = x
        attention_mask[i, pad_tokens:] = 1

    reference_model = DeltaMLP(
        **model_kwargs,
        use_padding_free_transformer=False,
    ).to(device=device, dtype=dtype)
    packed_model = DeltaMLP(
        **model_kwargs,
        use_padding_free_transformer=True,
    ).to(device=device, dtype=dtype)

    reference_model.eval()
    packed_model.eval()
    packed_model.load_state_dict(reference_model.state_dict())

    cu_seqlens = torch.tensor([0] + lengths, dtype=torch.int32, device=device)
    cu_seqlens = cu_seqlens.cumsum(dim=0)
    packed_input = torch.cat(inputs, dim=0)
    torch.testing.assert_close(padded_input[attention_mask.bool()], packed_input)

    monkeypatch.setattr(
        delta_mlp_module,
        "fused_recurrent_delta_rule",
        lambda *args, **kwargs: pytest.fail("packed DeltaMLP must use chunk mode"),
    )

    with torch.no_grad():
        expected_output = reference_model(
            padded_input,
            attention_mask=attention_mask,
        )
        expected_output = expected_output[attention_mask.bool()]

        packed_output = packed_model(
            packed_input,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

    torch.testing.assert_close(packed_output, expected_output)
