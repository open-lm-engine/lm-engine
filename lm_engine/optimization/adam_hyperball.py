# **************************************************
# Copyright (c) 2026, Mayank Mishra
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
    from xma import adam_hyperball


# compile makes a single graph which is very useful when we are using DTensors
@torch.compile
def _foreach_normalize(x_list: list[torch.Tensor], eps: float) -> None:
    u = torch._foreach_norm(x_list, dtype=torch.float32)
    torch._foreach_add_(u, eps)
    torch._foreach_div_(x_list, u)


class AdamHyperball(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-10,
        weight_decay: float = 0.1,
        hyperball: bool = False,
        maximize: bool = False,
    ) -> AdamHyperball:
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
            params: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            state_steps: list[torch.Tensor | int] = []

            if group["hyperball"]:
                Rs: list[torch.Tensor] = []

                self._init_adam_hyperball_group(
                    group=group,
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    Rs=Rs,
                    state_steps=state_steps,
                )

                if is_kernel_allowed(Kernel.adam_hyperball):
                    adam_hyperball(
                        params=params,
                        grads=grads,
                        exp_avgs=exp_avgs,
                        exp_avg_sqs=exp_avg_sqs,
                        Rs=Rs,
                        lr=group["lr"],
                        beta1=beta1,
                        beta2=beta2,
                        maximize=group["maximize"],
                        state_steps=state_steps,
                        eps=group["eps"],
                    )
                else:
                    eps = group["eps"]
                    lr = group["lr"]
                    if group["maximize"]:
                        lr = -lr

                    # update momentum
                    torch._foreach_mul_(exp_avgs, beta1)
                    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

                    # update variance
                    torch._foreach_mul_(exp_avg_sqs, beta2)
                    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

                    # get copy of variables to prevent updating inplace accidentaly
                    exp_avgs = torch._foreach_mul(exp_avgs, [1 / (1 - beta1**t) for t in state_steps])
                    exp_avg_sqs = torch._foreach_mul(exp_avg_sqs, [1 / (1 - beta2**t) for t in state_steps])

                    # compute Adam update
                    torch._foreach_sqrt_(exp_avg_sqs)
                    torch._foreach_add_(exp_avg_sqs, eps)
                    torch._foreach_div_(exp_avgs, exp_avg_sqs)

                    # normalize the Adam update
                    _foreach_normalize(x_list=exp_avgs, eps=eps)

                    # update the parameter
                    lr_Rs = torch._foreach_mul(Rs, lr)
                    torch._foreach_mul_(exp_avgs, lr_Rs)
                    torch._foreach_sub_(params, exp_avgs)

                    # normalize the updated parameter
                    _foreach_normalize(x_list=params, eps=eps)

                    # project parameters on hyperball of radius R
                    torch._foreach_mul_(params, Rs)
            else:
                max_exp_avg_sqs: list[torch.Tensor] = []

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

    def _init_adam_hyperball_group(
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
