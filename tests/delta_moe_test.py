# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Smoke tests for DeltaMoE: just checks that forward + backward run to completion and
produce finite output, for both the plain-MoE (delta_mlp=None) and delta_mlp-enabled paths.
Not a numerical correctness check."""

import pytest
import torch

from lm_engine.training.modeling_utils.mlp_blocks.delta_mlp.config import DeltaMLPArgs
from lm_engine.training.modeling_utils.mlp_blocks.delta_moe.config import DeltaMoEArgs
from lm_engine.training.modeling_utils.mlp_blocks.delta_moe.module import DeltaMoE
from lm_engine.utils import is_fla_available


_HIDDEN_SIZE = 32


def _delta_mlp_args(**overrides) -> DeltaMLPArgs:
    kwargs = dict(
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
        allow_neg_eigval=False,
        kernel_size=4,
    )
    kwargs.update(overrides)
    return DeltaMLPArgs(**kwargs)


def _make_deltamoe(delta_mlp: DeltaMLPArgs | None) -> DeltaMoE:
    config = DeltaMoEArgs(
        intermediate_size=64,
        activation_function="swiglu",
        add_bias=False,
        dropout=0,
        num_experts=4,
        num_experts_per_tok=2,
        delta_mlp=delta_mlp,
    )

    return DeltaMoE(
        hidden_size=_HIDDEN_SIZE,
        config=config,
        init_method="normal",
        initializer_range=0.02,
        m_width=None,
        num_layers=1,
        use_depth_scaled_init=False,
        norm_eps=1e-6,
        layer_idx=0,
        use_padding_free_transformer=False,
    )


def test_smoke_deltamoe_without_delta_mlp() -> None:
    """delta_mlp=None: plain MoE routing/experts, no delta-rule dependency (runs anywhere)."""
    model = _make_deltamoe(delta_mlp=None)

    x = torch.randn(2, 7, _HIDDEN_SIZE, requires_grad=True)
    out = model(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available() or not is_fla_available(), reason="CUDA and FLA are required")
def test_smoke_deltamoe_with_delta_mlp() -> None:
    """delta_mlp populated: exercises the DeltaMLP branch (chunk_delta_rule), which needs
    CUDA + FLA + triton. batch_size must be 1 in training mode (DeltaMLP's own dense-layout
    constraint)."""
    model = _make_deltamoe(delta_mlp=_delta_mlp_args()).to(device="cuda")

    x = torch.randn(1, 7, _HIDDEN_SIZE, device="cuda", requires_grad=True)
    out = model(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert torch.isfinite(x.grad).all()
