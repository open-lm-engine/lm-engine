# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Smoke tests for MuonHyperball: just checks that .step() runs to completion and actually
updates the parameter, for every dispatch path in MuonHyperball.step(). Not a numerical
correctness check (see the manual verification in the session that added these paths)."""

import torch

from lm_engine.training.optimization.muon_hyperball import MuonHyperball
from lm_engine.training.parameter import mark_parameter_as_attention_parameter, mark_parameter_as_conv_hyperball


def _run_step_and_assert_changed(opt: MuonHyperball, params: list[torch.nn.Parameter]) -> None:
    before = [p.detach().clone() for p in params]
    opt.step()

    for b, p in zip(before, params):
        assert not torch.allclose(b, p.detach())


def test_smoke_adamw_step() -> None:
    """non-hyperball param groups fall through to the plain AdamW path."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    p.grad = torch.randn(8, 8)

    opt = MuonHyperball([{"params": [p], "hyperball": False}])
    _run_step_and_assert_changed(opt, [p])


def test_smoke_muon_hyperball_step() -> None:
    """plain hyperball param groups (no conv/attention marking) go through _muon_hyperball_step."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    p.grad = torch.randn(8, 8)

    opt = MuonHyperball([{"params": [p], "hyperball": True}])
    _run_step_and_assert_changed(opt, [p])


def test_smoke_sgd_hyperball_step_conv() -> None:
    """conv_hyperball_group param groups go through _sgd_hyperball_step_conv."""
    p = torch.nn.Parameter(torch.randn(4, 3, 3, 3))
    mark_parameter_as_conv_hyperball(p)
    p.grad = torch.randn(4, 3, 3, 3)

    opt = MuonHyperball([{"params": [p], "hyperball": True, "conv_hyperball_group": True}])
    _run_step_and_assert_changed(opt, [p])


def test_smoke_muon_hyperball_step_attention() -> None:
    """attention_hyperball_group param groups (with split_attention_heads) go through
    _muon_hyperball_step_attention."""
    p = torch.nn.Parameter(torch.randn(16, 16))
    mark_parameter_as_attention_parameter(p, head_dim=4)
    p.grad = torch.randn(16, 16)

    opt = MuonHyperball(
        [{"params": [p], "hyperball": True, "attention_hyperball_group": True, "split_attention_heads": True}]
    )
    _run_step_and_assert_changed(opt, [p])


def test_smoke_multiple_steps() -> None:
    """a few consecutive steps across every path, in one optimizer (exercises momentum buffers
    and cached R across steps, not just the first/lazy-init step)."""
    p_normal = torch.nn.Parameter(torch.randn(8, 8))
    p_hyperball = torch.nn.Parameter(torch.randn(8, 8))
    p_conv = torch.nn.Parameter(torch.randn(4, 3, 3, 3))
    mark_parameter_as_conv_hyperball(p_conv)
    p_attn = torch.nn.Parameter(torch.randn(16, 16))
    mark_parameter_as_attention_parameter(p_attn, head_dim=4)

    opt = MuonHyperball(
        [
            {"params": [p_normal], "hyperball": False},
            {"params": [p_hyperball], "hyperball": True},
            {"params": [p_conv], "hyperball": True, "conv_hyperball_group": True},
            {
                "params": [p_attn],
                "hyperball": True,
                "attention_hyperball_group": True,
                "split_attention_heads": True,
            },
        ]
    )

    for _ in range(3):
        for p in [p_normal, p_hyperball, p_conv, p_attn]:
            p.grad = torch.randn_like(p)

        opt.step()

    for p in [p_normal, p_hyperball, p_conv, p_attn]:
        assert torch.isfinite(p.detach()).all()
