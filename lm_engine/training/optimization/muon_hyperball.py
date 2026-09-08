# **************************************************
# Copyright (c) 2026, Mayank Mishra
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
    """Per-row L2 normalization (per-row path), input shape [rows, k].

    Replaces NS for params where each row is an independent unit — conv kernels (one row per
    output channel) and 2D weights marked per-row-hyperball (e.g. DeltaMLP b_proj, one row per head).
    """
    assert G.dim() == 2
    return G / G.norm(dim=-1, keepdim=True).clamp(min=1e-7)


@torch.compile
def zeropower_via_newtonschulz(
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

    Accepts either a single matrix (2D, shape `[M, K]`) or a stack of same-shape matrices
    (3D, shape `[N, M, K]`). The 3D path uses batched matmul (`@` is bmm for ndim==3) so all
    N orthogonalizations run as a single launch per matmul, with much better tensor-core
    utilization than N sequential 2D calls.

    Adapted from https://github.com/KellerJordan/Muon/blob/master/muon.py
    """

    # FIXME jyo: this might come back to bite us
    # FIXME jyo: remove all .item() calls everywhere
    assert G.dim() in (2, 3)

    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-1, -2)
    # Per-(batch-)matrix Frobenius norm. For 2D this returns a 0-D tensor; for 3D it returns
    # `[N, 1, 1]` so the divide broadcasts back over each matrix in the batch.
    if X.dim() == 2:
        norm = X.norm()
    else:
        norm = X.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
    X = X / (norm + 1e-7)

    for steps, (a, b, c) in steps_and_coefficients:
        # Ensure spectral norm is at most 1
        # Perform the NS iterations
        for _ in range(steps):
            A = X @ X.transpose(-1, -2)
            B = b * A + c * A @ A
            X = a * X + B @ X

    if transposed:
        X = X.transpose(-1, -2)

    return X


@torch.compile
def hyperball_project_per_row(p_2d: torch.Tensor, u_hat: torch.Tensor, R: torch.Tensor, lr: float) -> torch.Tensor:
    """Per-row hyperball projection (per-row path).

    Each row independently projected onto its own hypersphere with radius R[row].
    """
    w_candidate = p_2d - lr * R * u_hat
    w_norm = w_candidate.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return w_candidate * (R / w_norm)


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
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
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
        eps = group["eps"]

        if hybrid_ns:
            steps_and_coefficients = [(8, (3.4445, -4.7750, 2.0315)), (2, (2, -1.5, 0.5))]
        else:
            steps_and_coefficients = [(5, (3.4445, -4.7750, 2.0315))]

        # ----- Phase 1: collect per-param state and prepare local views for foreach -----
        params: list[torch.Tensor] = []
        is_dtensors: list[bool] = []
        orig_shapes: list[torch.Size] = []
        is_per_rows: list[bool] = []
        bufs: list[torch.Tensor] = []
        buf_locals: list[torch.Tensor] = []
        grad_locals: list[torch.Tensor] = []
        Rs: list = []  # float for the NS path, tensor [C,1] for the per-row path

        for p in group["params"]:
            if p.grad is None:
                continue

            g = p.grad
            is_dtensor = isinstance(p, DTensor)
            orig_shape = p.size()
            # Per-row path: each row L2-normed + hyperball-projected independently, instead of
            # NS + scalar hyperball. Set for conv kernels (one row per output channel) and for
            # 2D weights marked via mark_parameter_as_per_row_hyperball (e.g. DeltaMLP b_proj).
            is_per_row = getattr(p, "_per_row_hyperball", False)

            if g.ndim > 2:
                if isinstance(g, DTensor):
                    g = g.full_tensor()
                g = g.view(g.size(0), -1)

            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(g)
                if is_per_row:
                    # per-kernel radius: shape [C, 1] from [C, 1, k] -> [C, k]
                    p_full = p.full_tensor() if is_dtensor else p
                    state["R"] = p_full.reshape(-1, p_full.shape[-1]).norm(dim=-1, keepdim=True)
                else:
                    # Cache R as Python float so it can feed _foreach_*.ScalarList without per-step sync
                    R_tensor = p.norm()
                    if is_dtensor:
                        R_tensor = R_tensor.full_tensor()
                    state["R"] = float(R_tensor.item())

            R_val = state["R"]
            # Backward-compat: if R was previously stored as a 0-d tensor (e.g. from old checkpoints),
            # convert once to a Python float for the NS path.
            if not is_per_row and isinstance(R_val, torch.Tensor):
                R_val = float(R_val.item())
                state["R"] = R_val

            buf = state["momentum_buffer"]
            # get the local shard of the DTensor - no communication is needed
            buf_local = buf.to_local() if isinstance(buf, DTensor) else buf
            g_local = g.to_local() if isinstance(g, DTensor) else g

            params.append(p)
            is_dtensors.append(is_dtensor)
            orig_shapes.append(orig_shape)
            is_per_rows.append(is_per_row)
            bufs.append(buf)
            buf_locals.append(buf_local)
            grad_locals.append(g_local)
            Rs.append(R_val)

        if not params:
            return

        # ----- Phase 2: foreach momentum update on local shards -----
        # Elementwise, so safe to operate per-shard at any world_size.
        torch._foreach_mul_(buf_locals, momentum)
        torch._foreach_add_(buf_locals, grad_locals)

        # ----- Phase 3: gather g_nes per param, then batched NS by shape -----
        # Per-param NS is matmul-based (no _foreach_matmul), but same-shape matrices stack into a
        # 3D tensor and run as a single bmm. With ~5-10 unique shapes and many params per shape,
        # this collapses N×15 sequential matmul launches into ~5-10 × 15 batched-matmul launches.
        g_nes_fulls: list[torch.Tensor] = [None] * len(params)
        for i, p in enumerate(params):
            buf = bufs[i]
            if isinstance(buf, DTensor):
                # ndim==2 DTensor case: compute g_nes on sharded DTensors then gather once
                if nesterov:
                    g_nes_fulls[i] = (p.grad + momentum * buf).full_tensor()
                else:
                    g_nes_fulls[i] = buf.full_tensor()
            else:
                # ndim>2 case (already gathered upfront) or plain tensor case
                if nesterov:
                    g_nes_fulls[i] = grad_locals[i] + momentum * buf
                else:
                    g_nes_fulls[i] = buf

        u_fulls: list[torch.Tensor] = [None] * len(params)

        # Group NS-path params by shape; each shape group runs as one batched NS call
        shape_to_idxs: dict[torch.Size, list[int]] = {}
        for i, sc in enumerate(is_per_rows):
            if sc:
                continue
            shape_to_idxs.setdefault(g_nes_fulls[i].shape, []).append(i)

        for shape, idxs in shape_to_idxs.items():
            if len(idxs) == 1:
                # Solo shape: skip the stack/unbind round-trip
                i = idxs[0]
                u_fulls[i] = zeropower_via_newtonschulz(g_nes_fulls[i], steps_and_coefficients=steps_and_coefficients)
            else:
                G_stack = torch.stack([g_nes_fulls[i] for i in idxs], dim=0)
                U_stack = zeropower_via_newtonschulz(G_stack, steps_and_coefficients=steps_and_coefficients)
                for j, i in enumerate(idxs):
                    u_fulls[i] = U_stack[j]

        # Per-row params: row-wise L2 norm (kernel_norm) instead of NS
        for i, sc in enumerate(is_per_rows):
            if not sc:
                continue
            u_fulls[i] = kernel_norm(g_nes_fulls[i])

        # ----- Phase 4: foreach normalize NS outputs (per-row params already normalized) -----
        ns_idxs = [i for i, sc in enumerate(is_per_rows) if not sc]
        if ns_idxs:
            ns_us = [u_fulls[i] for i in ns_idxs]
            ns_norms = torch._foreach_norm(ns_us)
            # Stack to a single 1-D tensor and convert with one CPU/GPU sync (rather than N)
            ns_norm_floats = torch.stack(ns_norms).tolist()
            ns_inv = [1.0 / (nf + eps) for nf in ns_norm_floats]
            torch._foreach_mul_(ns_us, ns_inv)

        # ----- Phase 5: foreach hyperball projection (NS-params) -----
        if ns_idxs:
            ns_ps_full: list[torch.Tensor] = []
            for i in ns_idxs:
                p = params[i]
                p_full = p.full_tensor() if is_dtensors[i] else p
                if len(orig_shapes[i]) > 2:
                    p_full = p_full.view(p.size(0), -1)
                ns_ps_full.append(p_full)

            ns_us = [u_fulls[i] for i in ns_idxs]
            ns_Rs = [Rs[i] for i in ns_idxs]

            # scaled_u[i] = lr * R[i] * u_hat[i] (per-tensor Python-float scalar)
            lr_R_floats = [lr * R for R in ns_Rs]
            scaled_us = torch._foreach_mul(ns_us, lr_R_floats)

            # w_candidate = p - scaled_u
            w_candidates = torch._foreach_sub(ns_ps_full, scaled_us)

            # final scale = R / (||w_candidate|| + eps)
            w_norms = torch._foreach_norm(w_candidates)
            w_norm_floats = torch.stack(w_norms).tolist()
            scale_floats = [R / (wn + eps) for R, wn in zip(ns_Rs, w_norm_floats)]
            torch._foreach_mul_(w_candidates, scale_floats)

            # Reshape + redistribute + copy back are necessarily per-param
            for j, i in enumerate(ns_idxs):
                p = params[i]
                w_new = w_candidates[j]
                if len(orig_shapes[i]) > 2:
                    w_new = w_new.view(orig_shapes[i])
                if is_dtensors[i]:
                    w_new = tensor_to_dtensor(
                        w_new,
                        device_mesh=p.device_mesh,
                        current_placement=[Replicate()] * len(p.placements),
                        desired_placement=p.placements,
                    )
                p.copy_(w_new)

        # ----- Phase 6: per-row params (per-param, small count, R is per-row tensor) -----
        for i, sc in enumerate(is_per_rows):
            if not sc:
                continue
            p = params[i]
            u_hat = u_fulls[i]
            R = Rs[i]
            p_full = p.full_tensor() if is_dtensors[i] else p
            p_2d = p_full.reshape(-1, orig_shapes[i][-1])
            w_new = hyperball_project_per_row(p_2d, u_hat, R, lr).reshape(orig_shapes[i])
            if is_dtensors[i]:
                w_new = tensor_to_dtensor(
                    w_new,
                    device_mesh=p.device_mesh,
                    current_placement=[Replicate()] * len(p.placements),
                    desired_placement=p.placements,
                )
            p.copy_(w_new)
