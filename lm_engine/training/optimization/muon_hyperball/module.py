# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed._tensor.api import DTensor
from torch.distributed.tensor import distribute_tensor
from torch.optim import AdamW, Optimizer
from torch.optim.adam import adam

from ...parameter import get_attention_head_dim, is_attention_parameter
from ..adam_hyperball.op import _foreach_normalize
from .utils import _get_newtonschulz_coefficients, _update_momentum_and_apply_nesterov, _zeropower_via_newtonschulz


_CHUNK_SIZE = 16


class MuonHyperball(Optimizer):
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
        mode: str = "ns_only",
        split_attention_heads: bool = True,
        normalize_grad_before_momentum: bool = False,
    ) -> MuonHyperball:
        assert mode in ["ns_only", "full"]

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
            mode=mode,
            split_attention_heads=split_attention_heads,
            normalize_grad_before_momentum=normalize_grad_before_momentum,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            if group.get("hyperball", False):
                if group.get("conv_hyperball_group", False):
                    self._sgd_hyperball_step_conv(group)
                elif group.get("attention_hyperball_group", False) and group.get("split_attention_heads", False):
                    self._muon_hyperball_step_attention(group)
                else:
                    self._muon_hyperball_step(group)
            else:
                self._adamw_step(group)

    def _adamw_step(self, group: dict) -> None:
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

    def _sgd_hyperball_step_conv(self, group: dict) -> None:
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        lr = group["lr"]
        eps = group["eps"]

        params, grads, momentum_buffer_list, Rs = self._init_hyperball_group(group, per_row_norm=True)

        if not params:
            return

        if momentum != 0:
            grads = _update_momentum_and_apply_nesterov(
                grads=grads, momentum_buffer_list=momentum_buffer_list, momentum=momentum, nesterov=nesterov
            )

        original_params = params

        params = [W.flatten(1) for W in params]
        grads = [dW.flatten(1) for dW in grads]

        grad_norms = [dW.float().norm(dim=-1, keepdim=True) for dW in grads]
        torch._foreach_add_(grad_norms, eps)
        grads = torch._foreach_div(grads, grad_norms)

        lr_Rs = torch._foreach_mul(Rs, lr)
        torch._foreach_mul_(grads, lr_Rs)
        torch._foreach_sub_(params, grads)

        param_norms = [W.float().norm(dim=-1, keepdim=True) for W in params]
        torch._foreach_add_(param_norms, eps)
        params = torch._foreach_div_(params, param_norms)
        torch._foreach_mul_(params, Rs)

        for W, _W in zip(original_params, params):
            W.copy_(_W.view_as(W))

    def _muon_hyperball_step(self, group: dict) -> None:
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        lr = group["lr"]
        eps = group["eps"]

        params, grads, momentum_buffer_list, Rs = self._init_hyperball_group(group, per_row_norm=False)

        if not params:
            return

        if group.get("normalize_grad_before_momentum", False):
            assert momentum != 0
            grads = _foreach_normalize(grads, eps=eps, in_place=False)

        if momentum != 0:
            grads = _update_momentum_and_apply_nesterov(
                grads=grads, momentum_buffer_list=momentum_buffer_list, momentum=momentum, nesterov=nesterov
            )

        steps_and_coefficients = _get_newtonschulz_coefficients(group.get("hybrid_ns", False))

        for chunk_start in range(0, len(grads), _CHUNK_SIZE):
            chunk = slice(chunk_start, chunk_start + _CHUNK_SIZE)
            params_chunk = params[chunk]
            Rs_chunk = Rs[chunk]

            ns_grads_chunk: list[torch.Tensor] = []
            for dW in grads[chunk]:
                is_dtensor = isinstance(dW, DTensor)

                _dW = dW.full_tensor() if is_dtensor else dW
                _dW = _zeropower_via_newtonschulz(_dW, steps_and_coefficients=steps_and_coefficients)

                if is_dtensor:
                    _dW = distribute_tensor(_dW, device_mesh=dW.device_mesh, placements=dW.placements)

                ns_grads_chunk.append(_dW)

            # normalize the Muon update
            _foreach_normalize(ns_grads_chunk, eps=eps)

            # update the parameter
            lr_Rs_chunk = torch._foreach_mul(Rs_chunk, lr)
            torch._foreach_mul_(ns_grads_chunk, lr_Rs_chunk)
            torch._foreach_sub_(params_chunk, ns_grads_chunk)

            # normalize the updated parameter
            _foreach_normalize(x_list=params_chunk, eps=eps)

            # project parameters on hyperball of radius R
            torch._foreach_mul_(params_chunk, Rs_chunk)

    def _muon_hyperball_step_attention(self, group: dict) -> None:
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        lr = group["lr"]
        eps = group["eps"]

        params, grads, momentum_buffer_list, Rs = self._init_hyperball_group(group, split_attention_heads=True)

        if not params:
            return

        if group.get("normalize_grad_before_momentum", False):
            assert momentum != 0
            grads = _foreach_normalize(grads, eps=eps, in_place=False)

        if momentum != 0:
            grads = _update_momentum_and_apply_nesterov(
                grads=grads, momentum_buffer_list=momentum_buffer_list, momentum=momentum, nesterov=nesterov
            )

        steps_and_coefficients = _get_newtonschulz_coefficients(group.get("hybrid_ns", False))

        for chunk_start in range(0, len(params), _CHUNK_SIZE):
            chunk = slice(chunk_start, chunk_start + _CHUNK_SIZE)

            params_chunk = params[chunk]
            Ds_chunk = [W.size(-1) for W in params_chunk]
            Hs_chunk = [get_attention_head_dim(W) for W in params_chunk]
            Rs_chunk = [R[:, None, None] for R in Rs[chunk]]

            dWs_chunk: list[torch.Tensor] = []
            for dW, H in zip(grads[chunk], Hs_chunk):
                _dW = dW.full_tensor() if isinstance(dW, DTensor) else dW
                _dW = _dW.reshape(-1, H, _dW.size(-1))
                dWs_chunk.append(_zeropower_via_newtonschulz(_dW, steps_and_coefficients=steps_and_coefficients))

            u_norms_chunk = [dW.float().flatten(-2).norm(dim=-1)[:, None, None] for dW in dWs_chunk]
            torch._foreach_add_(u_norms_chunk, eps)
            dWs_chunk = torch._foreach_div(dWs_chunk, u_norms_chunk)

            lr_Rs_chunk = torch._foreach_mul(Rs_chunk, lr)
            torch._foreach_mul_(dWs_chunk, lr_Rs_chunk)

            Ws_chunk = [
                (W.full_tensor() if isinstance(W, DTensor) else W).reshape(-1, H, D)
                for W, H, D in zip(params_chunk, Hs_chunk, Ds_chunk)
            ]
            torch._foreach_sub_(Ws_chunk, dWs_chunk)

            w_norms_chunk = [W.float().flatten(-2).norm(dim=-1)[:, None, None] for W in Ws_chunk]
            torch._foreach_add_(w_norms_chunk, eps)
            rescale_chunk = torch._foreach_div(Rs_chunk, w_norms_chunk)
            torch._foreach_mul_(Ws_chunk, rescale_chunk)

            for W, W_new, D in zip(params_chunk, Ws_chunk, Ds_chunk):
                W_new = W_new.reshape(-1, D)

                if isinstance(W, DTensor):
                    W_new = distribute_tensor(W_new, device_mesh=W.device_mesh, placements=W.placements)

                W.copy_(W_new)

    def _init_hyperball_group(
        self, group: dict, per_row_norm: bool = False, split_attention_heads: bool = False
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        params: list[torch.Tensor] = []
        momentum_buffer_list: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        Rs: list[torch.Tensor] = []

        for W in group["params"]:
            W: torch.Tensor
            dW: torch.Tensor | None = W.grad

            if dW is None:
                continue

            state = self.state[W]
            if len(state) == 0:
                if group["momentum"] != 0:
                    state["momentum_buffer"] = torch.zeros_like(W)

                H = None
                if split_attention_heads and is_attention_parameter(W):
                    H = get_attention_head_dim(W)

                if H is not None:
                    if isinstance(W, DTensor):
                        W = W.full_tensor()

                    R = W.reshape(-1, H, W.size(-1)).float().flatten(-2).norm(dim=-1)
                elif per_row_norm:
                    R = W.flatten(1).float().norm(dim=-1, keepdim=True)
                else:
                    R = W.float().norm()

                if isinstance(R, DTensor):
                    R = R.full_tensor()

                state["R"] = R

            params.append(W)
            grads.append(dW)
            momentum_buffer_list.append(state.get("momentum_buffer"))
            Rs.append(state["R"])

        return params, grads, momentum_buffer_list, Rs
