# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch
from einops import repeat

import lm_engine.training.modeling_utils.mlp_blocks.delta_mlp.module as delta_mlp_module
from lm_engine.training.enums import Kernel
from lm_engine.training.generation_cache import ConstantCache, GenerationCache, GenerationState, LinearCache
from lm_engine.training.kernels import enable_kernels
from lm_engine.training.modeling_utils import AttentionMaskInfo, DeltaMLP, DeltaMLPArgs
from lm_engine.utils import is_causal_conv1d_available, is_fla_available


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_fla_available(),
    reason="CUDA and FLA are required",
)

if is_fla_available():
    from lm_engine.training.modeling_utils.mlp_blocks.delta_mlp.utils import chunk_delta_rule


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
    config_kwargs = dict(
        intermediate_size=64,
        activation_function="silu",
        add_bias=False,
        dropout=0,
        num_ranks=8,
        num_heads=4,
        use_v_proj=False,
        use_q_l2norm=True,
        use_shortconv=False,
        use_tied_beta=False,
        use_decay_beta=False,
        use_o_norm=False,
        allow_neg_eigval=False,
        kernel_size=4,
        A_init_min=0,
        A_init_max=16,
        dt_init_min=0.001,
        dt_init_max=0.1,
        dt_init_floor=1e-4,
        value_scale=None,
    )
    kwargs = dict(
        hidden_size=32,
        layer_idx=0,
        norm_eps=1e-6,
        init_method="normal",
        initializer_range=0.02,
        m_width=None,
        num_layers=1,
        use_depth_scaled_init=False,
        sequence_parallel=False,
    )
    for key, value in overrides.items():
        if key in config_kwargs:
            config_kwargs[key] = value
        else:
            kwargs[key] = value

    kwargs["config"] = DeltaMLPArgs(**config_kwargs)
    return kwargs


def _cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([0] + lengths, dtype=torch.int32, device=device).cumsum(dim=0)


def _packed_one_shot_suffix_output(
    model: DeltaMLP,
    inputs: list[torch.Tensor],
    prefix_lengths: list[int],
) -> torch.Tensor:
    outputs = []
    for x, prefix_length in zip(inputs, prefix_lengths):
        output = model(
            x,
            attention_mask_info=AttentionMaskInfo(
                cu_seqlens=_cu_seqlens([x.size(0)], device=x.device), max_seqlen=x.size(0)
            ),
        )
        outputs.append(output[prefix_length:])

    return torch.cat(outputs, dim=0)


# Shared scaffolding for the packed-cache tests below. They all build one eval
# packed model, draw random per-request inputs, compute a one-shot reference,
# then replay it through the cache. Expected and cached runs share the same
# model (the reference is computed before any monkeypatch), so one instance
# suffices -- no separate reference/packed pair.
_PACKED_DEVICE = torch.device("cuda")
_PACKED_DTYPE = torch.bfloat16


def _packed_model(use_shortconv: bool) -> DeltaMLP:
    model = DeltaMLP(**_delta_mlp_kwargs(use_shortconv=use_shortconv), use_padding_free_transformer=True)
    return model.to(device=_PACKED_DEVICE, dtype=_PACKED_DTYPE).eval()


def _random_inputs(lengths: list[int], hidden_size: int) -> list[torch.Tensor]:
    return [torch.randn(length, hidden_size, dtype=_PACKED_DTYPE, device=_PACKED_DEVICE) for length in lengths]


def _forbid_fused_recurrent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        delta_mlp_module,
        "fused_recurrent_delta_rule",
        lambda *args, **kwargs: pytest.fail("packed DeltaMLP prefill must use chunk mode"),
    )


def _split_prefix_suffix(inputs: list[torch.Tensor], prefix_lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = torch.cat([x[:prefix_length] for x, prefix_length in zip(inputs, prefix_lengths)], dim=0)
    suffix = torch.cat([x[prefix_length:] for x, prefix_length in zip(inputs, prefix_lengths)], dim=0)
    return prefix, suffix


@pytest.mark.parametrize(
    "H, HV",
    [
        (1, 1),
        (4, 4),
        (8, 8),
        (4, 1),
        (8, 1),
        (5, 1),
        (7, 1),
        (4, 2),
        (8, 2),
        (8, 4),
        (6, 2),
        (6, 3),
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
def test_broadcast(H: int, HV: int, dtype: torch.dtype) -> None:
    B, T, DK, DV = 7, 1024, 128, 64
    assert H % HV == 0
    q, k, v, b, S = prepare_data(
        B=B,
        T=T,
        H=H,
        DK=DK,
        DV=DV,
        HK=H,
        HV=HV,
        HB=H,
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
        beta=b0.float(),
        initial_state=S0,
        output_final_state=True,
        cu_seqlens=None,
        use_q_l2norm_in_kernel=True,
        use_k_l2norm_in_kernel=True,
    )

    q1 = _leaf(q)
    k1 = _leaf(k)
    v1 = _leaf(v)
    b1 = _leaf(b)
    S1 = _leaf(S)

    v1_broadcast = repeat(v1, "... h d -> ... (h g) d", g=H // HV)
    S1_broadcast = repeat(S1, "1 ... -> b ...", b=B)

    o1, S1_ = chunk_delta_rule(
        q=q1,
        k=k1,
        v=v1_broadcast,
        beta=b1.float(),
        initial_state=S1_broadcast,
        output_final_state=True,
        cu_seqlens=None,
        use_q_l2norm_in_kernel=True,
        use_k_l2norm_in_kernel=True,
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
            attention_mask_info=AttentionMaskInfo(attention_mask=attention_mask),
        )
        expected_output = expected_output[attention_mask.bool()]

        packed_output = packed_model(
            packed_input,
            attention_mask_info=AttentionMaskInfo(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen),
        )

    torch.testing.assert_close(packed_output, expected_output)


@pytest.mark.parametrize("use_shortconv", [False, True])
@pytest.mark.parametrize(
    ("lengths", "prefix_lengths", "atol"),
    [
        # All-prefixed, split on a chunk boundary -> exact.
        pytest.param([97], [64], None, id="single_boundary"),
        pytest.param([97, 65], [64, 64], None, id="multi_boundary"),
        # All-prefixed, non-boundary split -> bf16 chunk-repartition drift.
        pytest.param([97, 65], [60, 40], 6e-2, id="non_boundary"),
        pytest.param([33, 17], [20, 10], 6e-2, id="short_chunk"),
        pytest.param([66, 67], [64, 64], 6e-2, id="suffix_below_kernel"),
        # Mixed fresh (prefix 0) + prefixed requests in one batch.
        pytest.param([33, 97], [0, 64], None, id="fresh_then_prefixed"),
        pytest.param([97, 33], [64, 0], None, id="prefixed_then_fresh"),
        pytest.param([33, 97, 40], [0, 64, 0], None, id="fresh_prefixed_fresh"),
    ],
)
def test_packed_cache_continuation(
    use_shortconv: bool,
    lengths: list[int],
    prefix_lengths: list[int],
    atol: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1234)
    device = _PACKED_DEVICE
    model = _packed_model(use_shortconv)
    suffix_lengths = [length - prefix for length, prefix in zip(lengths, prefix_lengths)]
    prefixed_idx = [i for i, prefix in enumerate(prefix_lengths) if prefix > 0]
    has_fresh = len(prefixed_idx) < len(lengths)
    inputs = _random_inputs(lengths, model.hidden_size)

    with torch.no_grad():
        expected_output = _packed_one_shot_suffix_output(model, inputs, prefix_lengths)

    _forbid_fused_recurrent(monkeypatch)

    cache = GenerationCache()
    with torch.no_grad():
        # Prefill the prefixed requests (all of them when none are fresh).
        prefix_input = torch.cat([inputs[i][: prefix_lengths[i]] for i in prefixed_idx], dim=0)
        model(
            prefix_input,
            cache_params=cache,
            attention_mask_info=AttentionMaskInfo(
                cu_seqlens=_cu_seqlens([prefix_lengths[i] for i in prefixed_idx], device=device),
                max_seqlen=max(prefix_lengths[i] for i in prefixed_idx),
            ),
        )
        conv_state, recurrent_state = cache.get_cache(layer_idx=0, empty_value=(None, None), cache_name="delta_mlp")

        if has_fresh:
            # Rebuild a full-batch state: prefilled state scattered into the
            # prefixed rows; fresh rows keep a zero conv + initial recurrent state.
            mixed_recurrent_state = model.initial_recurrent_state(batch_size=len(lengths))
            assert recurrent_state is not None
            for seeded, i in enumerate(prefixed_idx):
                mixed_recurrent_state[i] = recurrent_state[seeded]

            if use_shortconv:
                assert conv_state is not None
                mixed_conv_state = torch.zeros(
                    len(lengths), *conv_state.shape[1:], dtype=conv_state.dtype, device=conv_state.device
                )
                for seeded, i in enumerate(prefixed_idx):
                    mixed_conv_state[i] = conv_state[seeded]
            else:
                mixed_conv_state = None

            cache.update(
                states=(
                    GenerationState(state=mixed_conv_state, method=ConstantCache),
                    GenerationState(state=mixed_recurrent_state, method=ConstantCache),
                ),
                layer_idx=0,
                cache_name="delta_mlp",
            )
        else:
            # Direct packed prefill must write one state row per request.
            assert recurrent_state is not None and recurrent_state.shape[0] == len(lengths)
            if use_shortconv:
                assert conv_state is not None and conv_state.shape[0] == len(lengths)
            else:
                assert conv_state is None

        _, suffix_input = _split_prefix_suffix(inputs, prefix_lengths)
        actual_output = model(
            suffix_input,
            cache_params=cache,
            attention_mask_info=AttentionMaskInfo(
                cu_seqlens=_cu_seqlens(suffix_lengths, device=device), max_seqlen=max(suffix_lengths)
            ),
        )

    if atol is None:
        torch.testing.assert_close(actual_output, expected_output)
    else:
        torch.testing.assert_close(actual_output, expected_output, rtol=2e-2, atol=atol)


@pytest.mark.parametrize("use_shortconv", [False, True])
@pytest.mark.parametrize(
    ("lengths", "prefix_lengths", "atol"),
    [
        pytest.param([67, 131], [64, 128], 2e-2, id="chunk_boundary"),
        pytest.param([63, 65], [60, 62], 6e-2, id="non_boundary"),
    ],
)
def test_packed_cache_decode_loop(
    use_shortconv: bool,
    lengths: list[int],
    prefix_lengths: list[int],
    atol: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1234)
    device = _PACKED_DEVICE
    model = _packed_model(use_shortconv)
    suffix_lengths = [length - prefix_length for length, prefix_length in zip(lengths, prefix_lengths)]
    assert len(set(suffix_lengths)) == 1
    decode_steps = suffix_lengths[0]
    inputs = _random_inputs(lengths, model.hidden_size)

    with torch.no_grad():
        expected_suffixes = [
            model(
                x,
                attention_mask_info=AttentionMaskInfo(
                    cu_seqlens=_cu_seqlens([x.size(0)], device=device), max_seqlen=x.size(0)
                ),
            )[prefix_length:]
            for x, prefix_length in zip(inputs, prefix_lengths)
        ]

    fused_recurrent_delta_rule = delta_mlp_module.fused_recurrent_delta_rule
    fused_recurrent_calls = 0

    def wrapped_fused_recurrent_delta_rule(*args, **kwargs):
        nonlocal fused_recurrent_calls
        fused_recurrent_calls += 1
        return fused_recurrent_delta_rule(*args, **kwargs)

    monkeypatch.setattr(
        delta_mlp_module,
        "fused_recurrent_delta_rule",
        wrapped_fused_recurrent_delta_rule,
    )

    prefix_input, _ = _split_prefix_suffix(inputs, prefix_lengths)

    cache = GenerationCache()
    with torch.no_grad():
        model(
            prefix_input,
            cache_params=cache,
            attention_mask_info=AttentionMaskInfo(
                cu_seqlens=_cu_seqlens(prefix_lengths, device=device), max_seqlen=max(prefix_lengths)
            ),
        )
        for step in range(decode_steps):
            actual_output = model(
                torch.cat(
                    [
                        x[prefix_length + step : prefix_length + step + 1]
                        for x, prefix_length in zip(inputs, prefix_lengths)
                    ],
                    dim=0,
                ),
                cache_params=cache,
                attention_mask_info=AttentionMaskInfo(
                    cu_seqlens=_cu_seqlens([1] * len(lengths), device=device), max_seqlen=1
                ),
            )
            expected_output = torch.cat([suffix[step : step + 1] for suffix in expected_suffixes], dim=0)

            # Decode intentionally switches each one-token suffix to fused
            # recurrent, while one-shot ground truth uses chunk for the full
            # sequence. The per-case atol covers this kernel difference plus
            # three steps of cached-state drift; increase it deliberately if
            # this test starts decoding more steps.
            torch.testing.assert_close(actual_output, expected_output, rtol=2e-2, atol=atol)

    assert fused_recurrent_calls == decode_steps


def _fresh_zero_state_cache(model: DeltaMLP, batch_size: int) -> GenerationCache:
    """Cache holding a zero conv state + learned initial recurrent state -- how
    serving represents a batch of entirely fresh requests (conv fresh == zero,
    recurrent fresh == learned initial state)."""
    param = next(model.parameters())
    conv_state = (
        torch.zeros(
            batch_size,
            model.kv_conv1d.in_channels,
            model.kv_conv1d.kernel_size,
            dtype=param.dtype,
            device=param.device,
        )
        if model.use_shortconv
        else None
    )
    cache = GenerationCache()
    cache.update(
        states=(
            GenerationState(state=conv_state, method=ConstantCache),
            GenerationState(state=model.initial_recurrent_state(batch_size=batch_size), method=ConstantCache),
        ),
        layer_idx=0,
        cache_name="delta_mlp",
    )
    return cache


@pytest.mark.parametrize("use_shortconv", [False, True])
@pytest.mark.parametrize(
    ("lengths", "decode"),
    [
        pytest.param([24], False, id="prefill_single"),
        pytest.param([33, 17], False, id="prefill_multi"),
        pytest.param([24, 24, 24], False, id="prefill_three"),
        pytest.param([1, 1], True, id="decode"),
    ],
)
def test_packed_cache_zero_state(
    use_shortconv: bool,
    lengths: list[int],
    decode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Serving encodes fresh requests as a ZERO conv state (never None), so they
    # ride the cached path alongside prefixed ones. This pins that path
    # (the conv_state=None equivalence tests never reach it) and the
    # zero == no-history contract. The decode case additionally exercises the
    # unit-length causal_conv1d_update kernel.
    if decode and use_shortconv and not is_causal_conv1d_available():
        pytest.skip("causal_conv1d kernel required for the decode update path")

    torch.manual_seed(1234)
    dtype = torch.bfloat16
    device = torch.device("cuda")
    model = DeltaMLP(**_delta_mlp_kwargs(use_shortconv=use_shortconv), use_padding_free_transformer=True).to(
        device=device, dtype=dtype
    )
    model.eval()

    inputs = [torch.randn(length, model.hidden_size, dtype=dtype, device=device) for length in lengths]

    with torch.no_grad():
        expected_output = _packed_one_shot_suffix_output(model, inputs, prefix_lengths=[0] * len(lengths))

    if not decode:
        # Prefill must use chunk mode; decode intentionally uses fused recurrent.
        monkeypatch.setattr(
            delta_mlp_module,
            "fused_recurrent_delta_rule",
            lambda *args, **kwargs: pytest.fail("packed prefill must use chunk mode"),
        )

    cache = _fresh_zero_state_cache(model, batch_size=len(lengths))
    with enable_kernels([Kernel.causal_conv1d] if decode else []), torch.no_grad():
        actual_output = model(
            torch.cat(inputs, dim=0),
            cache_params=cache,
            attention_mask_info=AttentionMaskInfo(
                cu_seqlens=_cu_seqlens(lengths, device=device), max_seqlen=max(lengths)
            ),
        )

    # Same math as one-shot, but the bf16 conv kernel and chunk partition differ
    # from a per-sequence one-shot run, so allow a small tolerance.
    torch.testing.assert_close(actual_output, expected_output, rtol=2e-2, atol=6e-2)


def test_generation_cache() -> None:
    cache = GenerationCache()
    key = torch.randn(1, 2, 3)
    value = torch.randn(1, 2, 3)
    conv_state = torch.randn(1, 4)
    recurrent_state = torch.randn(1, 5)

    cache.update(
        states=(
            GenerationState(state=key, method=LinearCache),
            GenerationState(state=value, method=LinearCache),
        ),
        layer_idx=0,
    )
    cache.update(
        states=(
            GenerationState(state=conv_state, method=ConstantCache),
            GenerationState(state=recurrent_state, method=ConstantCache),
        ),
        layer_idx=0,
        cache_name="delta_mlp",
    )

    actual_key, actual_value = cache.get_cache(layer_idx=0, empty_value=(None, None))
    actual_conv_state, actual_recurrent_state = cache.get_cache(
        layer_idx=0,
        empty_value=(None, None),
        cache_name="delta_mlp",
    )
    torch.testing.assert_close(actual_key, key)
    torch.testing.assert_close(actual_value, value)
    torch.testing.assert_close(actual_conv_state, conv_state)
    torch.testing.assert_close(actual_recurrent_state, recurrent_state)
    assert cache.get_seq_length(layer_idx=0) == key.size(1)
    assert cache.get_seq_length(layer_idx=0, cache_name="delta_mlp") == 0
