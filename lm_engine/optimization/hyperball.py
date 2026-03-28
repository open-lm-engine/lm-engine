# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from torch.optim import Optimizer


class HyperballAdamW(Optimizer):
    """Hyperball optimizer with AdamW fallback for non-projection parameters.

    For parameter groups with hyperball=True (projection weight matrices):
        Constrains weights to lie on a hypersphere of fixed radius R = ||W_0||_F.
        Uses Adam to compute the update direction u_t, then applies:
            W_{t+1} = R * Normalize(W_t - lr * R * Normalize(u_t))

    For all other parameter groups:
        Standard AdamW update with optional weight decay.

    Args:
        params: iterable of parameters or param groups
        lr: learning rate (eta in the Hyperball formula)
        betas: coefficients for computing running averages of gradient and its square
        eps: term added to denominator for numerical stability
        weight_decay: weight decay coefficient (applied only to non-hyperball groups)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-10,
        weight_decay: float = 0.1,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hyperball=False)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("hyperball", False):
                self._hyperball_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _hyperball_step(self, group: dict) -> None:
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]

            # initialize hyperball on the first step
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["R"] = p.norm().item()

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            t = state["step"]
            R = state["R"]

            self._update_moments(exp_avg=exp_avg, exp_avg_sq=exp_avg_sq, grad=p.grad, beta1=beta1, beta2=beta2)

            # Bias-corrected Adam update direction
            bc1 = 1 - beta1**t
            bc2 = 1 - beta2**t
            u_t = (exp_avg / bc1) / ((exp_avg_sq / bc2).sqrt_().add_(eps))

            # Normalize update direction
            u_norm = u_t.norm()
            # TODO mayank fix this CPU side check
            if u_norm == 0:
                continue
            u_hat = u_t / u_norm

            # Step on the sphere surface, then project back
            w_candidate = p - lr * R * u_hat
            w_norm = w_candidate.norm()
            # TODO mayank fix this CPU side check
            if w_norm == 0:
                continue
            p.copy_(w_candidate.mul_(R / w_norm))

    def _adamw_step(self, group: dict) -> None:
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            t = state["step"]

            # Weight decay (decoupled)
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)

            self._update_moments(exp_avg=exp_avg, exp_avg_sq=exp_avg_sq, grad=p.grad, beta1=beta1, beta2=beta2)

            # Bias-corrected update
            bc1 = 1 - beta1**t
            bc2 = 1 - beta2**t
            step_size = lr / bc1
            denom = (exp_avg_sq / bc2).sqrt_().add_(eps)

            p.addcdiv_(exp_avg, denom, value=-step_size)

    def _update_moments(
        self, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor, grad: torch.Tensor, beta1: float, beta2: float
    ) -> None:
        # Adam moments
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
