# **************************************************
# Copyright (c) 2026, Mayank Mishra, Jyo Pari, Zhonglin Han
# **************************************************

from __future__ import annotations

from typing import Callable

import torch
from torch.distributed.tensor import DTensor
from torch.optim import AdamW, Optimizer
from torch.optim.adam import adam

from ..enums import Kernel
from ..kernels import is_kernel_allowed
from ..utils import is_xma_available


if is_xma_available():
    from xma import hyperball_adam


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
        hyperball: bool = False,
        maximize: bool = False,
    ) -> HyperballAdamW:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hyperball=hyperball,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
            amsgrad=False,
            maximize=maximize,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            if group["hyperball"]:
                params = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                Rs = []
                state_steps = []

                self._init_hyperball_group(
                    group=group,
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    Rs=Rs,
                    state_steps=state_steps,
                )

                if is_kernel_allowed(Kernel.hyperball_adam):
                    hyperball_adam(
                        params=params,
                        grads=grads,
                        exp_avgs=exp_avgs,
                        exp_avg_sqs=exp_avg_sqs,
                        Rs=Rs,
                        lr=group["lr"],
                        beta1=beta1,
                        beta2=beta2,
                        maximize=group["maximize"],
                        foreach=group["foreach"],
                        state_steps=state_steps,
                        eps=group["eps"],
                    )
                else:
                    eps = group["eps"]
                    lr = group["lr"]
                    if group["maximize"]:
                        lr = -lr

                    for i, (W, dW, exp_avg, exp_avg_sq, t, R) in enumerate(
                        zip(params, grads, exp_avgs, exp_avg_sqs, state_steps, Rs)
                    ):
                        exp_avg.mul_(beta1).add_(dW, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(dW, dW, value=1 - beta2)

                        bc1 = 1 / (1 - beta1**t)
                        bc2 = 1 / (1 - beta2**t)
                        u = exp_avg * bc1 / (exp_avg_sq * bc2).sqrt_().add_(eps)

                        # Normalize update direction
                        u /= u.norm() + eps

                        # Step on the sphere surface, then project back
                        u *= lr * R
                        W -= u
                        W /= W.norm() + eps
                        W *= R

                        state_steps[i] += 1
            else:
                params: list[torch.Tensor] = []
                grads: list[torch.Tensor] = []
                exp_avgs: list[torch.Tensor] = []
                exp_avg_sqs: list[torch.Tensor] = []
                max_exp_avg_sqs: list[torch.Tensor] = []
                state_steps: list[torch.Tensor] = []

                has_complex = AdamW._init_group(
                    self,
                    group=group,
                    params_with_grad=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    max_exp_avg_sqs=max_exp_avg_sqs,
                    state_steps=state_steps,
                )

                adam(
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    max_exp_avg_sqs=max_exp_avg_sqs,
                    state_steps=state_steps,
                    amsgrad=group["amsgrad"],
                    has_complex=has_complex,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=group["maximize"],
                    foreach=group["foreach"],
                    capturable=group["capturable"],
                    differentiable=group["differentiable"],
                    fused=group["fused"],
                    grad_scale=getattr(self, "grad_scale", None),
                    found_inf=getattr(self, "found_inf", None),
                    decoupled_weight_decay=True,
                )

        return loss

    def _init_hyperball_group(
        self,
        group: dict,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        Rs: list[torch.Tensor],
        state_steps: list[int],
    ) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]

            if len(state) == 0:
                state["step"] = 1
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

                # do the communication for R ahead of time to prevent it on every timestep
                R = p.norm()
                if isinstance(R, DTensor):
                    R = R.full_tensor()
                state["R"] = R

            params.append(p)
            grads.append(p.grad)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            Rs.append(state["R"])
            state_steps.append(state["step"])
