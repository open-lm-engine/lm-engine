# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch
from torch.optim import AdamW

from lm_engine.optimization.adam_hyperball import HyperballAdamW


LR = 0.01
BETAS = (0.9, 0.95)
EPS = 1e-10
WEIGHT_DECAY = 0.1


@pytest.mark.parametrize(
    "param_data,grad_sequences",
    [
        ([3.0, 4.0], [[0.1, -0.2]]),
        ([3.0, 4.0], [[0.1 * i, -0.2 * i] for i in range(1, 6)]),
        ([[3.0, 4.0], [0.0, 5.0]], [[[0.1, -0.2], [0.3, -0.1]] for _ in range(3)]),
    ],
)
def test_hyperball(param_data, grad_sequences):
    p = torch.tensor(param_data, dtype=torch.float32)
    opt = HyperballAdamW([{"params": [p], "hyperball": True, "weight_decay": 0}], lr=LR, betas=BETAS, eps=EPS)

    W_ref = torch.tensor(param_data, dtype=torch.float32)
    opt_ref = AdamW([W_ref], lr=LR, betas=BETAS, eps=EPS, weight_decay=0.0)
    R = W_ref.data.norm().item()

    for grad_data in grad_sequences:
        grad = torch.tensor(grad_data, dtype=torch.float32)

        p.grad = grad.clone()
        opt.step()

        W_before = W_ref.data.clone()
        W_ref.grad = grad.clone()
        opt_ref.step()
        u_t = (W_before - W_ref.data) / LR

        u_hat = u_t / u_t.norm()
        w_cand = W_before - LR * R * u_hat
        W_expected = R * w_cand / w_cand.norm()

        W_ref.data.copy_(W_expected)
        torch.testing.assert_close(p.data, W_expected, atol=1e-5, rtol=1e-5)


def test_norm_preserved() -> None:
    p = torch.tensor([3.0, 4.0], dtype=torch.float32)
    R = p.data.norm().item()
    opt = HyperballAdamW([{"params": [p], "hyperball": True, "weight_decay": 0}], lr=LR, betas=BETAS, eps=EPS)

    for _ in range(20):
        p.grad = torch.randn_like(p.data)
        opt.step()
        assert abs(p.data.norm().item() - R) < 1e-5


def test_zero_grad_skipped() -> None:
    p = torch.tensor([3.0, 4.0], dtype=torch.float32)
    W_before = p.data.clone()
    opt = HyperballAdamW([{"params": [p], "hyperball": True, "weight_decay": 0}], lr=LR, betas=BETAS, eps=EPS)

    p.grad = torch.zeros_like(p.data)
    opt.step()

    torch.testing.assert_close(p.data, W_before)
