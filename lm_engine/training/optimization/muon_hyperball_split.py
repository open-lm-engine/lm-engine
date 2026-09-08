# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed._tensor.api import DTensor
from torch.distributed.tensor import Replicate

from ..dtensors import tensor_to_dtensor
from .muon_hyperball import MuonHyperball, hyperball_project_per_row, kernel_norm, zeropower_via_newtonschulz
from .muon_split_utils import active_spec, log_ns_batches, reverse_pattern, verify_and_log_specs


class MuonHSplit(MuonHyperball):
    """Standalone Muon variant driven by einops patterns from the config.

    Each splittable parameter is tagged via ..hf_models.parameter.set_split_spec
    with (tag, shape). The optimizer's `patterns` config maps each tag to an
    einops "lhs -> rhs" string plus the axis lengths that factor the LHS.

    `targets` selects which tags are active. `mode` is "ns_only" (split drives only
    NS) or "full" (split also drives per-batch normalize + hyperball projection
    with per-batch radii)."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        targets: list[str] | None = None,
        patterns: dict[str, dict] | None = None,
        mode: str = "ns_only",
        normalize_grad_before_momentum: bool = False,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        if mode not in ("ns_only", "full"):
            raise ValueError(f"mode must be 'ns_only' or 'full', got {mode!r}")
        self.targets = set(targets or [])
        self.patterns = dict(patterns or {})
        self.mode = mode
        self.normalize_grad_before_momentum = normalize_grad_before_momentum
        self._logged_ns_batches = False  # NS-batch composition logged once on first step
        for group in self.param_groups:
            for p, name in zip(group["params"], group.pop("param_names", [])):
                p._debug_name = name
        verify_and_log_specs(self.param_groups, self.targets, self.patterns, self.mode)

    def log_routing_table(self) -> None:
        from .muon_split_utils import log_param_routing_table

        named_params = [
            (p._debug_name, p, group.get("hyperball", False)) for group in self.param_groups for p in group["params"]
        ]
        log_param_routing_table(named_params, self.targets, self.patterns)

    def _muon_hyperball_step(self, group: dict) -> None:
        from einops import rearrange

        momentum = group["momentum"]
        nesterov = group["nesterov"]
        hybrid_ns = group.get("hybrid_ns", False)
        lr = group["lr"]
        eps = group["eps"]

        if hybrid_ns:
            steps_and_coefficients = [(8, (3.4445, -4.7750, 2.0315)), (2, (2, -1.5, 0.5))]
        else:
            steps_and_coefficients = [(5, (3.4445, -4.7750, 2.0315))]

        # Validate parameter/gradient distributed layouts before touching gradients.
        for p in group["params"]:
            if p.grad is None:
                continue

            is_param_dtensor = isinstance(p, DTensor)
            is_grad_dtensor = isinstance(p.grad, DTensor)

            if is_param_dtensor != is_grad_dtensor:
                raise RuntimeError(
                    f"Parameter/gradient DTensor mismatch for "
                    f"{getattr(p, '_debug_name', '<unnamed>')}: "
                    f"parameter is DTensor={is_param_dtensor}, "
                    f"gradient is DTensor={is_grad_dtensor}"
                )

        # optionally normalize each grad to unit Frobenius norm before momentum (fp32 norm, per-param)
        if self.normalize_grad_before_momentum:
            local_grads: list[torch.Tensor] = []
            norms: list[torch.Tensor] = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                gn = g.norm(dtype=torch.float32)
                if isinstance(gn, DTensor):
                    gn = gn.full_tensor()  # reduce to the true global norm (to_local gives per-shard)
                local_grads.append(g.to_local() if isinstance(g, DTensor) else g)
                norms.append(gn)
            if local_grads:
                scales = 1.0 / (torch.stack(norms) + eps)
                torch._foreach_mul_(local_grads, list(scales.unbind()))

        # Phase 1 — gather per-param state into parallel column-major lists so we can drive
        # the rest of the step with batched foreach ops. Lazy-init momentum_buffer and R on
        # first encounter; R lives as scalar (ns_only path), per-batch tensor (full path on
        # spec'd params), or per-row tensor (per-row params).
        params: list[torch.Tensor] = []
        is_dtensors: list[bool] = []
        orig_shapes: list[torch.Size] = []
        is_per_rows: list[bool] = []
        bufs: list[torch.Tensor] = []
        buf_locals: list[torch.Tensor] = []
        grad_locals: list[torch.Tensor] = []
        Rs: list = []
        specs: list = []

        for p in group["params"]:
            if p.grad is None:
                continue

            g = p.grad

            is_dtensor = isinstance(p, DTensor)
            is_grad_dtensor = isinstance(g, DTensor)

            if is_dtensor != is_grad_dtensor:
                raise RuntimeError(
                    f"Parameter/gradient DTensor mismatch for "
                    f"{getattr(p, '_debug_name', '<unnamed>')}: "
                    f"parameter is DTensor={is_dtensor}, "
                    f"gradient is DTensor={is_grad_dtensor}"
                )

            orig_shape = p.size()
            # Per-row path: each row L2-normed + hyperball-projected independently, instead of
            # NS + scalar hyperball. Set for conv kernels (one row per output channel) and for
            # 2D weights marked via mark_parameter_as_per_row_hyperball (e.g. DeltaMLP b_proj).
            is_per_row = getattr(p, "_per_row_hyperball", False)

            if g.ndim > 2:
                if isinstance(g, DTensor):
                    g = g.full_tensor()
                g = g.view(g.size(0), -1)

            spec = None if is_per_row else active_spec(p, self.targets, self.patterns)
            use_per_batch_R = spec is not None and self.mode == "full"

            # Lazy state init (first step for this param): momentum buffer + the fixed hyperball
            # radius R = ||W_0||, computed at the granularity this param is routed at.
            # full_tensor() gathers sharded DTensors so the norm covers the whole weight.
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(g)
                if is_per_row:
                    # one radius per row (rows, 1) — conv channels / b_proj heads
                    p_full = p.full_tensor() if is_dtensor else p
                    state["R"] = p_full.reshape(-1, p_full.shape[-1]).norm(dim=-1, keepdim=True)
                elif use_per_batch_R:
                    # one radius per split-batch (num_batches,) — per-head, full mode only
                    pattern, axes = spec
                    p_full = p.full_tensor() if is_dtensor else p
                    p_3d = rearrange(p_full, pattern, **axes)
                    state["R"] = p_3d.flatten(-2).norm(dim=-1).detach()
                else:
                    # one scalar radius for the whole matrix — plain NS / ns_only split
                    R_tensor = p.norm()
                    if is_dtensor:
                        R_tensor = R_tensor.full_tensor()
                    state["R"] = float(R_tensor.item())

            R_val = state["R"]
            if not is_per_row and not use_per_batch_R and isinstance(R_val, torch.Tensor):
                R_val = float(R_val.item())
                state["R"] = R_val

            buf = state["momentum_buffer"]
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
            specs.append(spec)

        if not params:
            return

        # Phase 2 — momentum-buffer update on local shards, batched: buf = momentum·buf + grad.
        torch._foreach_mul_(buf_locals, momentum)
        torch._foreach_add_(buf_locals, grad_locals)

        # Phase 3 — build nesterov-adjusted gradient per param on FULL (unsharded) tensors.
        # NS needs the full matrix to compute a polar factor, so DTensors get unsharded here.
        g_nes_fulls: list[torch.Tensor] = [None] * len(params)
        for i, p in enumerate(params):
            buf = bufs[i]
            if isinstance(buf, DTensor):
                if nesterov:
                    g_nes_fulls[i] = (p.grad + momentum * buf).full_tensor()
                else:
                    g_nes_fulls[i] = buf.full_tensor()
            else:
                if nesterov:
                    g_nes_fulls[i] = grad_locals[i] + momentum * buf
                else:
                    g_nes_fulls[i] = buf

        u_fulls: list[torch.Tensor] = [None] * len(params)

        # Phase 4 — bucket params for batched NS by routing path:
        #   spec_groups       : spec'd + targeted → 3D batched NS on rearranged-per-head form
        #   plain_shape_groups: NS-path, non-spec'd → 2D batched NS keyed by raw shape
        #   per-row           : conv kernels / b_proj → kernel_norm (no NS), handled below
        spec_groups: dict = {}
        plain_idxs: list[int] = []
        for i, sc in enumerate(is_per_rows):
            if sc:
                continue
            if specs[i] is not None:
                pattern, axes = specs[i]
                key = (pattern, tuple(sorted(axes.items())))
                spec_groups.setdefault(key, ([], specs[i]))[0].append(i)
            else:
                plain_idxs.append(i)

        # Collect (label, NS-input shape, member names) on the first step only, for logging.
        ns_batch_log: list[tuple[str, tuple, list[str]]] | None = [] if not self._logged_ns_batches else None

        # Spec path: rearrange each grad to its (num_batches, M, K) per-head form, cat across
        # all params sharing the same (pattern, axes), run NS once batched, then unsplit.
        for key, (idxs, spec) in spec_groups.items():
            pattern, axes = spec
            stacks = [rearrange(g_nes_fulls[i], pattern, **axes) for i in idxs]
            per_param_batch = stacks[0].shape[0]
            G_cat = torch.cat(stacks, dim=0)
            U_cat = zeropower_via_newtonschulz(G_cat, steps_and_coefficients=steps_and_coefficients)
            reverse = reverse_pattern(pattern)
            for j, i in enumerate(idxs):
                u_3d = U_cat[j * per_param_batch : (j + 1) * per_param_batch]
                u_fulls[i] = rearrange(u_3d, reverse, **axes)
            if ns_batch_log is not None:
                ns_batch_log.append(
                    (f"spec[{pattern}]", tuple(G_cat.shape), [getattr(params[i], "_debug_name", "?") for i in idxs])
                )

        # Plain path: bucket by raw 2D shape, stack same-shape params, one batched NS per bucket.
        plain_shape_groups: dict[torch.Size, list[int]] = {}
        for i in plain_idxs:
            plain_shape_groups.setdefault(g_nes_fulls[i].shape, []).append(i)
        for idxs in plain_shape_groups.values():
            if len(idxs) == 1:
                i = idxs[0]
                u_fulls[i] = zeropower_via_newtonschulz(g_nes_fulls[i], steps_and_coefficients=steps_and_coefficients)
                ns_shape = tuple(g_nes_fulls[i].shape)
            else:
                G_stack = torch.stack([g_nes_fulls[i] for i in idxs], dim=0)
                U_stack = zeropower_via_newtonschulz(G_stack, steps_and_coefficients=steps_and_coefficients)
                for j, i in enumerate(idxs):
                    u_fulls[i] = U_stack[j]
                ns_shape = tuple(G_stack.shape)
            if ns_batch_log is not None:
                ns_batch_log.append(
                    (
                        f"plain{tuple(g_nes_fulls[idxs[0]].shape)}",
                        ns_shape,
                        [getattr(params[i], "_debug_name", "?") for i in idxs],
                    )
                )

        # Per-row path: row-wise L2 norm (kernel_norm) in place of NS.
        for i, sc in enumerate(is_per_rows):
            if not sc:
                continue
            u_fulls[i] = kernel_norm(g_nes_fulls[i])

        if ns_batch_log is not None:
            log_ns_batches(ns_batch_log)
            self._logged_ns_batches = True

        # Phase 5a — full mode only: re-split each spec'd u back to 3D, normalize PER-BATCH
        # (each head gets unit Frobenius), then hyperball-project PER-BATCH with per-head R.
        # This is the "per-head decoupling" that distinguishes full from ns_only mode.
        full_spec_idxs = (
            [i for i in range(len(params)) if not is_per_rows[i] and specs[i] is not None]
            if self.mode == "full"
            else []
        )

        for i in full_spec_idxs:
            pattern, axes = specs[i]
            reverse = reverse_pattern(pattern)
            u_3d = rearrange(u_fulls[i], pattern, **axes)
            u_norms = u_3d.flatten(-2).norm(dim=-1)
            u_3d = u_3d / (u_norms[:, None, None] + eps)

            p = params[i]
            p_full = p.full_tensor() if is_dtensors[i] else p
            p_3d = rearrange(p_full, pattern, **axes)

            R = Rs[i]
            lr_R = lr * R
            scaled_u = lr_R[:, None, None] * u_3d
            w_candidate = p_3d - scaled_u
            w_norm = w_candidate.flatten(-2).norm(dim=-1)
            final_scale = R / (w_norm + eps)
            w_new_3d = w_candidate * final_scale[:, None, None]
            w_new = rearrange(w_new_3d, reverse, **axes)

            if is_dtensors[i]:
                w_new = tensor_to_dtensor(
                    w_new,
                    device_mesh=p.device_mesh,
                    current_placement=[Replicate()] * len(p.placements),
                    desired_placement=p.placements,
                )
            p.copy_(w_new)

        # Phase 5b — every NS-path param not already handled above (i.e. plain in any mode,
        # or spec'd in ns_only mode): normalize u with a SINGLE scalar Frobenius norm, then
        # hyperball-project with a single scalar R. Done with foreach for batched efficiency.
        full_spec_set = set(full_spec_idxs)
        regular_idxs = [i for i in range(len(params)) if not is_per_rows[i] and i not in full_spec_set]

        if regular_idxs:
            ns_us = [u_fulls[i] for i in regular_idxs]
            ns_norms = torch._foreach_norm(ns_us)
            ns_norm_floats = torch.stack(ns_norms).tolist()
            ns_inv = [1.0 / (nf + eps) for nf in ns_norm_floats]
            torch._foreach_mul_(ns_us, ns_inv)

            ns_ps_full: list[torch.Tensor] = []
            for i in regular_idxs:
                p = params[i]
                p_full = p.full_tensor() if is_dtensors[i] else p
                if len(orig_shapes[i]) > 2:
                    p_full = p_full.view(p.size(0), -1)
                ns_ps_full.append(p_full)

            ns_Rs = [Rs[i] for i in regular_idxs]
            lr_R_floats = [lr * R for R in ns_Rs]
            scaled_us = torch._foreach_mul(ns_us, lr_R_floats)
            w_candidates = torch._foreach_sub(ns_ps_full, scaled_us)
            w_norms = torch._foreach_norm(w_candidates)
            w_norm_floats = torch.stack(w_norms).tolist()
            scale_floats = [R / (wn + eps) for R, wn in zip(ns_Rs, w_norm_floats)]
            torch._foreach_mul_(w_candidates, scale_floats)

            for j, i in enumerate(regular_idxs):
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

        # Phase 5c — per-row params: per-row hyperball projection (one R per row).
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
