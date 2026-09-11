# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch


def _update_momentum_and_apply_nesterov(
    grads: list[torch.Tensor], momentum_buffer_list: list[torch.Tensor], momentum: float, nesterov: bool
) -> list[torch.Tensor]:
    torch._foreach_mul_(momentum_buffer_list, momentum)
    torch._foreach_add_(momentum_buffer_list, grads)

    if nesterov:
        grads = torch._foreach_add(grads, momentum_buffer_list, alpha=momentum)
    else:
        grads = momentum_buffer_list

    return grads


def _get_newtonschulz_coefficients(
    hybrid: bool,
) -> tuple[tuple[int, tuple[float, float, float]], tuple[int, tuple[float, float, float]]]:
    if hybrid:
        steps_and_coefficients = [(8, (3.4445, -4.7750, 2.0315)), (2, (2, -1.5, 0.5))]
    else:
        steps_and_coefficients = [(5, (3.4445, -4.7750, 2.0315))]

    return steps_and_coefficients


@torch.compile
def _zeropower_via_newtonschulz(
    G: torch.Tensor, steps_and_coefficients: tuple[int, tuple[int, int, int]]
) -> torch.Tensor:
    assert G.dim() in (2, 3)

    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-1, -2)

    if X.dim() == 2:
        norm = X.norm()
    else:
        norm = X.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)

    norm_inv = 1 / (norm + 1e-7)
    X = X * norm_inv

    for steps, (a, b, c) in steps_and_coefficients:
        for _ in range(steps):
            A = X @ X.transpose(-1, -2)
            B = b * A + c * A @ A
            X = a * X + B @ X

    if transposed:
        X = X.transpose(-1, -2)

    return X
