# **************************************************
# Copyright (c) 2026, Jyo Pari
# **************************************************

from typing import Callable

import torch
from torch.distributed._tensor.api import DTensor
from torch.distributed.tensor import Replicate
from torch.optim import AdamW, Optimizer
from torch.optim.adam import adam

from ..dtensors import tensor_to_dtensor


@torch.compile
def kernel_norm(G: torch.Tensor) -> torch.Tensor:
    """Per-row L2 normalization for depthwise conv kernels, input shape [C, k]."""
    assert G.dim() == 2
    return G / G.norm(dim=-1, keepdim=True).clamp(min=1e-7)


@torch.compile
def zeropower_via_newtonschulz5(
    G: torch.Tensor, steps_and_coefficients: tuple[int, tuple[int, int, int]] = (5, (3.4445, -4.7750, 2.0315))
) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Adapted from https://github.com/KellerJordan/Muon/blob/master/muon.py
    """
    assert G.dim() == 2

    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)

    for steps, (a, b, c) in steps_and_coefficients:
        # Ensure spectral norm is at most 1
        # Perform the NS iterations
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X


class MuonHyperball(Optimizer):
    """Muon optimizer with Hyperball constraint for projection weights, AdamW for others.

    For parameter groups with hyperball=True (projection weight matrices):
        Constrains weights to lie on a hypersphere of fixed radius R = ||W_0||_F.
        Uses Muon (Newton-Schulz orthogonalized Nesterov momentum) to compute the
        update direction u_t, then applies the Hyperball sphere projection:
            W_{t+1} = R * Normalize(W_t - lr * R * Normalize(u_t))

    For all other parameter groups:
        Standard AdamW update with optional weight decay.

    ZeRO-3 compatibility:
        NS orthogonalization requires a full 2D matrix. For 2D DTensor params (FSDP-2
        ZeRO-3), the Nesterov gradient is all-gathered before NS via .full_tensor(), then
        redistributed back via distribute_tensor() before the hyperball projection.
        For 3D DTensor params (e.g. depthwise conv weights), both the gradient and the
        param are all-gathered and reshaped to 2D for the full step; the result is
        reshaped back to the original shape and copied back via p.copy_().

    Args:
        params: iterable of parameters or param groups
        lr: learning rate (step size on the hypersphere for hyperball params)
        momentum: SGD momentum coefficient for Muon's internal momentum buffer
        nesterov: whether to use Nesterov-style momentum (recommended)
        ns_steps: number of Newton-Schulz iterations (5 is sufficient)
        betas: (beta1, beta2) for the AdamW fallback on non-hyperball params
        eps: epsilon for AdamW numerical stability
        weight_decay: weight decay coefficient (applied only to non-hyperball groups)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
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
                self._muon_hyperball_step(group)
            else:
                params: list[torch.Tensor] = []
                grads: list[torch.Tensor] = []
                exp_avgs: list[torch.Tensor] = []
                exp_avg_sqs: list[torch.Tensor] = []
                max_exp_avg_sqs: list[torch.Tensor] = []
                state_steps: list[torch.Tensor] = []
                beta1, beta2 = group["betas"]

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

    def _muon_hyperball_step(self, group: dict) -> None:
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        hybrid_ns = group.get("hybrid_ns", False)
        lr = group["lr"]

        for p in group["params"]:
            if p.grad is None:
                continue

            g = p.grad
            is_dtensor = isinstance(p, DTensor)
            orig_shape = p.size()

            if g.ndim > 2:
                # DTensor view with ndim>2 is unsupported under ZeRO-3 sharding propagation;
                # all-gather first, then reshape to 2D on a plain tensor
                if isinstance(g, DTensor):
                    g = g.full_tensor()
                g = g.view(g.size(0), -1)

            state = self.state[p]

            is_shortconv = len(orig_shape) == 3

            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(g)
                if is_shortconv:
                    # per-kernel radius: shape [C, 1] from [C, 1, k] -> [C, k]
                    p_full = p.full_tensor() if is_dtensor else p
                    R = p_full.reshape(-1, p_full.shape[-1]).norm(dim=-1, keepdim=True)
                else:
                    # R is the fixed hypersphere radius — must be the global Frobenius norm
                    R = p.norm()
                    if is_dtensor:
                        R = R.full_tensor()
                state["R"] = R

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)

            if nesterov:
                g_nes = g.add(buf, alpha=momentum)
            else:
                g_nes = buf

            # all-gather from shards if still a DTensor
            # (for ndim>2 params g was already gathered above, so g_nes is a plain tensor)
            if isinstance(g_nes, DTensor):
                g_nes_full = g_nes.full_tensor()
            else:
                g_nes_full = g_nes

            R = state["R"]

            # TODO Mayank: write kernels for Muon Hyperball
            # TODO Mayank: add marker for indicating convolution weight

            if is_shortconv:
                # per-row L2 normalization instead of Newton-Schulz
                u_hat_full = kernel_norm(g_nes_full)

                # per-row hyperball projection
                p_2d = (p.full_tensor() if is_dtensor else p).reshape(-1, orig_shape[-1])
                w_candidate = p_2d - lr * R * u_hat_full
                w_norm = w_candidate.norm(dim=-1, keepdim=True).clamp_(min=1e-7)
                w_new = w_candidate.mul_(R / w_norm).reshape(orig_shape)
                if is_dtensor:
                    w_new = tensor_to_dtensor(
                        w_new,
                        device_mesh=p.device_mesh,
                        current_placement=[Replicate()] * len(p.placements),
                        desired_placement=p.placements,
                    )
                p.copy_(w_new)
            else:
                if hybrid_ns:
                    steps_and_coefficients = [(8, (3.4445, -4.7750, 2.0315)), (2, (2, -1.5, 0.5))]
                else:
                    steps_and_coefficients = [(5, (3.4445, -4.7750, 2.0315))]

                u_full = zeropower_via_newtonschulz5(g_nes_full, steps_and_coefficients=steps_and_coefficients)
                u_norm = u_full.norm() + group["eps"]
                u_hat_full = u_full / u_norm

                # Project onto hypersphere: W_{t+1} = R * Normalize(W_t - lr * R * u_hat)
                if is_dtensor and len(orig_shape) > 2:
                    # p is a 3D DTensor; gather and reshape to 2D so the projection stays in
                    # the same [out, -1] space as u_hat_full, then copy back reshaped
                    p_2d = p.full_tensor().view(p.size(0), -1)
                    w_candidate = p_2d - lr * R * u_hat_full
                    w_norm = w_candidate.norm() + group["eps"]
                    w_new = w_candidate.mul_(R / w_norm).view(orig_shape)
                    w_new = tensor_to_dtensor(
                        w_new,
                        device_mesh=p.device_mesh,
                        current_placement=[Replicate()] * len(p.placements),
                        desired_placement=p.placements,
                    )
                    p.copy_(w_new)
                else:
                    # Redistribute the normalized update direction back to match p's sharding
                    if is_dtensor:
                        u_hat = tensor_to_dtensor(
                            u_hat_full,
                            device_mesh=p.device_mesh,
                            current_placement=[Replicate()] * len(p.placements),
                            desired_placement=p.placements,
                        )
                    else:
                        u_hat = u_hat_full

                    w_candidate = p - lr * R * u_hat
                    w_norm = w_candidate.norm()
                    if is_dtensor:
                        w_norm = w_norm.full_tensor()
                    p.copy_(w_candidate.mul_(R / (w_norm + group["eps"])))
