# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
from einops import rearrange
from fla.modules.convolution import CausalConv1dFunctionCP as _FLACausalConv1dFunctionCP
from fla.ops.cp import FLACPContext, conv_cp_send_recv_fwd
from fla.utils import input_guard

from .....utils import is_causal_conv1d_available


if is_causal_conv1d_available():
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function, causal_conv1d_fwd_function


class CausalConv1dFunctionCP(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        cp_context: FLACPContext | None,
    ):
        assert x.dim() == 3 and x.shape[0] == 1, f"CP requires [1, T, D], got {x.shape}"
        if cp_context is None:
            raise ValueError("cp_context must be provided for CausalConv1dFunctionCP")
        group = cp_context.group
        W = weight.shape[-1]

        tails = x[0, -(W - 1) :, :].contiguous()
        if not cp_context.is_first_rank:
            heads = conv_cp_send_recv_fwd(tails, group)
            valid_len = min(W - 1, cp_context.pre_num_conv_tokens)
            assert heads.shape[0] == valid_len
            initial_state = rearrange(heads, "w d -> 1 d w")
        else:
            _ = conv_cp_send_recv_fwd(tails, group)
            initial_state = None

        silu_activation = activation in ("silu", "swish")
        y = causal_conv1d_fwd_function(
            x=rearrange(x, "b t d -> b d t"),
            weight=weight,
            bias=bias,
            seq_idx=None,
            initial_states=initial_state,
            final_states_out=None,
            silu_activation=silu_activation,
        )
        y = rearrange(y, "b d t -> b t d")
        ctx.save_for_backward(x, weight, bias, initial_state)
        ctx.silu_activation = silu_activation
        ctx.group = group
        ctx.W = W
        ctx.is_first_rank = cp_context.is_first_rank
        ctx.pre_num_conv_tokens = cp_context.pre_num_conv_tokens

        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor):
        x, weight, bias, initial_state = ctx.saved_tensors
        group = ctx.group
        W = ctx.W

        dx, dw, db, dh0 = causal_conv1d_bwd_function(
            x=rearrange(x, "b t d -> b d t"),
            weight=weight,
            bias=bias,
            dout=rearrange(dy, "b t d -> b d t"),
            seq_idx=None,
            initial_states=initial_state,
            dfinal_states=None,
            dx=None,
            return_dinitial_states=initial_state is not None,
            silu_activation=ctx.silu_activation,
        )
        dx = rearrange(dx, "b d t -> b t d")

        # Correct dx gradients for CP
        _FLACausalConv1dFunctionCP._correct_dx_for_cp(
            dx=dx,
            dh0=dh0,
            W=W,
            group=group,
            is_first_rank=ctx.is_first_rank,
            pre_num_conv_tokens=ctx.pre_num_conv_tokens,
        )

        return dx, dw, db, None, None


@torch.compiler.disable
def causal_conv1d_cp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cp_context: FLACPContext | None = None,
):
    if cp_context is None:
        raise ValueError("cp_context must be provided for causal_conv1d_cp")

    assert cp_context.conv1d_kernel_size is not None, "conv1d_kernel_size must be provided for causal_conv1d_cp"
    assert cp_context.cu_seqlens is not None
    assert len(cp_context.cu_seqlens) == 2
    return CausalConv1dFunctionCP.apply(
        x,
        weight,
        bias,
        activation,
        cp_context,
    )
