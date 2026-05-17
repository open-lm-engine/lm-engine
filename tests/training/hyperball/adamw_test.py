# **************************************************
# Copyright (c) 2026, Jyo Pari
# **************************************************

import pytest
import torch
from torch.optim import AdamW

from lm_engine.optimization.adam_hyperball import HyperballAdamW


LR = 0.01
BETAS = (0.9, 0.95)
EPS = 1e-10


@pytest.mark.parametrize(
    "param_data,grad_data",
    [
        ([1.0, 2.0, 3.0], [0.1, -0.2, 0.3]),
        ([[1.0, 2.0], [3.0, 4.0]], [[0.1, -0.2], [0.3, -0.4]]),
    ],
)
@pytest.mark.parametrize("steps", [1, 3, 5, 10])
@pytest.mark.parametrize("weight_decay", [0, 0.1])
def test_adamw(param_data, grad_data, steps, weight_decay):
    p_ours = torch.tensor(param_data, dtype=torch.float32)
    p_ref = torch.tensor(param_data, dtype=torch.float32)

    opt_ours = HyperballAdamW([p_ours], lr=LR, betas=BETAS, eps=EPS, weight_decay=weight_decay)
    opt_ref = AdamW([p_ref], lr=LR, betas=BETAS, eps=EPS, weight_decay=weight_decay)

    grads = [torch.tensor(grad_data, dtype=torch.float32) for _ in range(steps)]

    for g in grads:
        p_ours.grad = g.clone()
        p_ref.grad = g.clone()
        opt_ours.step()
        opt_ref.step()

    torch.testing.assert_close(p_ours, p_ref)
