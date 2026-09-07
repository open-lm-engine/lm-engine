# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .group_backward_kernel import group_bwd_W
from .group_kernel import group
from .scatter_kernel import scatter2scatter


class _UpProjectionExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
    ):
        output = torch.empty(sorted_expert_idxs.size(0), expert_weights.size(-1), device=x.device, dtype=x.dtype)

        scatter2scatter(
            X=x,
            W=expert_weights,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=output,
            FAN_OUT=k,
            x_grouped=False,
            y_grouped=True,
        )

        ctx.save_for_backward(x, expert_weights, sorted_expert_idxs, sorted_scattered_idxs, expert_offsets)
        ctx.k = k

        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, expert_weights, sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = ctx.saved_tensors
        k = ctx.k

        grouped_x = torch.empty(sorted_scattered_idxs.size(0), x.size(1), dtype=x.dtype, device=x.device)
        group(A=x, sorted_expert_idxs=sorted_scattered_idxs, out=grouped_x, fan_out=k)

        d_weights = torch.zeros_like(expert_weights)

        group_bwd_W(
            DY=grad_out,
            X=grouped_x,
            expert_offsets=expert_offsets,
            DW=d_weights,
            E=expert_weights.size(0),
        )

        scatter2scatter(
            X=grad_out,
            W=expert_weights.permute(0, 2, 1),
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=grouped_x,
            FAN_OUT=1,
            x_grouped=True,
            y_grouped=False,
        )

        d_input = grouped_x.view(x.size(0), k, grouped_x.size(-1)).sum(-2)

        return d_input, d_weights, None, None, None, None


class _DownProjectionExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        router_weights,
    ):
        output = torch.empty(sorted_expert_idxs.size(0), expert_weights.size(-1), device=x.device, dtype=x.dtype)

        scatter2scatter(
            X=x,
            W=expert_weights,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=output,
            FAN_OUT=k,
            x_grouped=True,
            y_grouped=False,
        )

        output_expanded = output.view(router_weights.size(0), router_weights.size(1), output.size(-1))
        output = torch.bmm(router_weights.unsqueeze(1), output_expanded).squeeze(1)

        ctx.save_for_backward(
            x, expert_weights, sorted_expert_idxs, sorted_scattered_idxs, expert_offsets, router_weights
        )
        ctx.k = k

        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, expert_weights, sorted_expert_idxs, sorted_scattered_idxs, expert_offsets, router_weights = (
            ctx.saved_tensors
        )
        k = ctx.k

        output = torch.empty(sorted_expert_idxs.size(0), expert_weights.size(-1), device=x.device, dtype=x.dtype)

        scatter2scatter(
            X=x,
            W=expert_weights,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=output,
            FAN_OUT=k,
            x_grouped=True,
            y_grouped=False,
        )

        output_expanded = output.view(router_weights.size(0), router_weights.size(1), output.size(-1))
        d_router_weights = torch.bmm(output_expanded, grad_out.unsqueeze(2)).squeeze(-1)
        grouped_grad_out = output_expanded.flatten(0, 1)

        group(
            A=grad_out,
            sorted_expert_idxs=sorted_scattered_idxs,
            out=grouped_grad_out,
            coeff=router_weights.flatten(),
            fan_out=router_weights.size(1),
        )

        d_weights = torch.zeros_like(expert_weights)
        group_bwd_W(DY=grouped_grad_out, X=x, expert_offsets=expert_offsets, DW=d_weights, E=expert_weights.size(0))

        d_expanded_input = torch.empty(
            sorted_expert_idxs.size(0), expert_weights.size(1), device=x.device, dtype=x.dtype
        )

        scatter2scatter(
            X=grouped_grad_out,
            W=expert_weights.permute(0, 2, 1),
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=d_expanded_input,
            FAN_OUT=1,
            x_grouped=True,
            y_grouped=True,
        )

        d_input = d_expanded_input.view(x.size(0), k, d_expanded_input.size(-1)).sum(-2)

        return d_input, d_weights, None, None, None, None, d_router_weights


def up_projection_experts(
    x,
    expert_weights,
    k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    expert_offsets,
):
    return _UpProjectionExperts.apply(
        x,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
    )


def down_projection_experts(
    x,
    expert_weights,
    k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    expert_offsets,
    router_weights,
):
    return _DownProjectionExperts.apply(
        x,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        router_weights,
    )
