# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DTensor, Partial, Replicate

from ....kernels.accelerator import KernelBackend, get_kernel_backend
from ....utils import is_triton_available


if is_triton_available():
    from .triton_implementation import (
        _single_tensor_hyperball_state_update_triton,
        _single_tensor_hyperball_weight_norm_triton,
        _single_tensor_hyperball_weight_update_triton,
    )


def _single_tensor_hyperball_state_update(
    W: torch.Tensor,
    dW: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    u_norm: torch.Tensor,
    beta1: float,
    beta2: float,
    t: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.empty_like(W)

    is_dtensor = isinstance(dW, DTensor)

    _single_tensor_hyperball_state_update_triton(
        exp_avg=exp_avg.to_local() if isinstance(exp_avg, DTensor) else exp_avg,
        exp_avg_sq=exp_avg_sq.to_local() if isinstance(exp_avg_sq, DTensor) else exp_avg_sq,
        dW=dW.to_local() if isinstance(dW, DTensor) else dW,
        u=u.to_local() if isinstance(u, DTensor) else u,
        u_norm=u_norm,
        beta1=beta1,
        beta2=beta2,
        t=t,
        eps=eps,
        sm_margin=int(is_dtensor),
    )

    if is_dtensor:
        u_norm = DTensor.from_local(
            u_norm,
            device_mesh=dW.device_mesh,
            placements=[Partial() if placement.is_shard() else placement for placement in dW.placements],
        ).redistribute(placements=[Replicate()] * dW.device_mesh.ndim, async_op=True)

    return u, u_norm


def _single_tensor_hyperball_weight_norm(
    W: torch.Tensor,
    u: torch.Tensor,
    u_norm: torch.Tensor,
    W_norm: torch.Tensor,
    lr: float,
    R: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if isinstance(u_norm, DTensor):
        u_norm = u_norm.to_local()
        if isinstance(u_norm, AsyncCollectiveTensor):
            u_norm = u_norm.wait()

    if isinstance(W_norm, DTensor):
        W_norm = W_norm.to_local()
        if isinstance(W_norm, AsyncCollectiveTensor):
            W_norm = W_norm.wait()

    is_dtensor = isinstance(W, DTensor)

    _single_tensor_hyperball_weight_norm_triton(
        u=u.to_local() if isinstance(u, DTensor) else u,
        u_norm=u_norm,
        lr=lr,
        R=R.to_local() if isinstance(R, DTensor) else R,
        W=W.to_local() if isinstance(W, DTensor) else W,
        W_norm=W_norm,
        eps=eps,
        sm_margin=int(is_dtensor),
    )

    if is_dtensor:
        W_norm = DTensor.from_local(
            W_norm,
            device_mesh=W.device_mesh,
            placements=[Partial() if placement.is_shard() else placement for placement in W.placements],
        ).redistribute(placements=[Replicate()] * W.device_mesh.ndim, async_op=True)

    return W_norm


def _single_tensor_hyperball_weight_update(
    W: torch.Tensor,
    u: torch.Tensor,
    u_norm: torch.Tensor,
    W_norm: torch.Tensor,
    lr: float,
    R: torch.Tensor,
    eps: float,
) -> None:
    if isinstance(u_norm, DTensor):
        u_norm = u_norm.to_local()
        if isinstance(u_norm, AsyncCollectiveTensor):
            u_norm = u_norm.wait()

    if isinstance(W_norm, DTensor):
        W_norm = W_norm.to_local()
        if isinstance(W_norm, AsyncCollectiveTensor):
            W_norm = W_norm.wait()

    _single_tensor_hyperball_weight_update_triton(
        u=u.to_local() if isinstance(u, DTensor) else u,
        u_norm=u_norm,
        lr=lr,
        R=R.to_local() if isinstance(R, DTensor) else R,
        W=W.to_local() if isinstance(W, DTensor) else W,
        W_norm=W_norm,
        eps=eps,
        sm_margin=int(isinstance(u, DTensor)),
    )


@torch.profiler.record_function("_single_tensor_adam_hyperball")
def _single_tensor_adam_hyperball(
    W0: torch.Tensor,
    W1: torch.Tensor | None,
    dW0: torch.Tensor,
    dW1: torch.Tensor | None,
    exp_avg0: torch.Tensor,
    exp_avg1: torch.Tensor | None,
    exp_avg_sq0: torch.Tensor,
    exp_avg_sq1: torch.Tensor | None,
    R0: torch.Tensor,
    R1: torch.Tensor | None,
    lr: float,
    beta1: float,
    beta2: float,
    t0: int,
    t1: int | None,
    eps: float | None = None,
) -> None:
    u_norm0, u_norm1, W_norm0, W_norm1 = torch.zeros((4,), dtype=torch.float32, device=W0.device).chunk(4)

    u0, u_norm0 = _single_tensor_hyperball_state_update(
        W=W0, dW=dW0, exp_avg=exp_avg0, exp_avg_sq=exp_avg_sq0, u_norm=u_norm0, beta1=beta1, beta2=beta2, t=t0, eps=eps
    )

    if W1 is not None:
        u1, u_norm1 = _single_tensor_hyperball_state_update(
            W=W1,
            dW=dW1,
            exp_avg=exp_avg1,
            exp_avg_sq=exp_avg_sq1,
            u_norm=u_norm1,
            beta1=beta1,
            beta2=beta2,
            t=t1,
            eps=eps,
        )

    W_norm0 = _single_tensor_hyperball_weight_norm(W=W0, u=u0, u_norm=u_norm0, W_norm=W_norm0, lr=lr, R=R0, eps=eps)

    if W1 is not None:
        W_norm1 = _single_tensor_hyperball_weight_norm(
            W=W1, u=u1, u_norm=u_norm1, W_norm=W_norm1, lr=lr, R=R1, eps=eps
        )

    _single_tensor_hyperball_weight_update(W=W0, u=u0, u_norm=u_norm0, W_norm=W_norm0, lr=lr, R=R0, eps=eps)

    if W1 is not None:
        _single_tensor_hyperball_weight_update(W=W1, u=u1, u_norm=u_norm1, W_norm=W_norm1, lr=lr, R=R1, eps=eps)


# compile makes a single graph which is very useful when we are using DTensors
@torch.compile
def _foreach_normalize(x_list: list[torch.Tensor], eps: float) -> None:
    u = torch._foreach_norm(x_list, dtype=torch.float32)
    torch._foreach_add_(u, eps)
    torch._foreach_div_(x_list, u)


@torch.no_grad()
def adam_hyperball(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    Rs: list[torch.Tensor],
    lr: float,
    beta1: float,
    beta2: float,
    maximize: bool,
    state_steps: list[int],
    eps: float | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> None:
    if kernel_backend is None:
        kernel_backend = get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    if maximize:
        lr = -lr

    if eps is None:
        eps = torch.finfo(params[0].dtype).eps

    if kernel_backend in [KernelBackend.cuda, KernelBackend.triton]:
        n = len(params)
        for i in range(0, n, 2):
            f = lambda x: None if i == n - 1 else x[i + 1]

            _single_tensor_adam_hyperball(
                W0=params[i],
                W1=f(params),
                dW0=grads[i],
                dW1=f(grads),
                exp_avg0=exp_avgs[i],
                exp_avg1=f(exp_avgs),
                exp_avg_sq0=exp_avg_sqs[i],
                exp_avg_sq1=f(exp_avg_sqs),
                R0=Rs[i],
                R1=f(Rs),
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                t0=state_steps[i],
                t1=f(state_steps),
                eps=eps,
            )
    elif kernel_backend == KernelBackend.torch:
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
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    for i in range(len(state_steps)):
        state_steps[i] += 1
