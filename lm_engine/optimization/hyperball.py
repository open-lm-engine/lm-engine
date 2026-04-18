# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from torch.distributed.tensor import DTensor
from torch.optim import AdamW, Optimizer
from torch.optim.adam import adam


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
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hyperball=False,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
            amsgrad=False,
        )

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
                # do the communication for R ahead of time to prevent it on every timestep
                R = p.norm()
                if isinstance(R, DTensor):
                    R = R.full_tensor()
                state["R"] = R

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
            u_norm = u_t.norm() + eps
            u_hat = u_t / u_norm

            # Step on the sphere surface, then project back
            w_candidate = p - lr * R * u_hat
            w_norm = w_candidate.norm() + eps
            p.copy_(w_candidate.mul_(R / w_norm))

    def _adamw_step(self, group: dict) -> None:
        params_with_grad: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        exp_avgs: list[torch.Tensor] = []
        exp_avg_sqs: list[torch.Tensor] = []
        max_exp_avg_sqs: list[torch.Tensor] = []
        state_steps: list[torch.Tensor] = []
        beta1, beta2 = group["betas"]

        has_complex = AdamW._init_group(
            self,
            group,
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        )

        adam(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=group["amsgrad"],
            has_complex=has_complex,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=False,
            foreach=group["foreach"],
            capturable=group["capturable"],
            differentiable=group["differentiable"],
            fused=group["fused"],
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None),
            decoupled_weight_decay=True,
        )

    def _update_moments(
        self, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor, grad: torch.Tensor, beta1: float, beta2: float
    ) -> None:
        # Adam moments
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
