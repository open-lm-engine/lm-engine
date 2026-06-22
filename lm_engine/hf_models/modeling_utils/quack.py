# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor

from ...utils import is_quack_available
from .activations import get_activation_function


if is_quack_available():
    from quack.gemm_interface import gemm, gemm_act, gemm_gated
    from quack.linear import linear_func, linear_fwd_convert_type


_QUACK_GEMM_ACT_MAPPING = {"gelu_pytorch_tanh": "gelu_tanh_approx", "relu": "relu", "silu": "silu"}
_QUACK_GEMM_GATED_MAPPING = {"glu": "glu", "reglu": "reglu", "sigmoid_glu": "glu", "swiglu": "swiglu"}


def _get_quack_activation(activation_function: str, mapping: dict[str, str], kernel_name: str) -> str:
    if activation_function not in mapping:
        raise ValueError(f"activation function ({activation_function}) is not supported by {kernel_name}")

    return mapping[activation_function]


def _is_supported(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> bool:
    tensors = [input, weight]
    if bias is not None:
        tensors.append(bias)

    return all(tensor.is_cuda and tensor.dtype in [torch.float16, torch.bfloat16] for tensor in tensors)


def quack_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor | None:
    if _is_supported(input, weight, bias):
        return linear_func(input, weight, bias=bias, tuned=False)

    return None


class _FusedMLPFC1ActFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation_function: str,
        quack_activation: str,
        is_glu: bool,
    ) -> torch.Tensor:
        x, weight = linear_fwd_convert_type(x, weight)

        with torch.amp.autocast("cuda", enabled=False):
            batch_shape = x.shape[:-1]
            x = x.flatten(0, -2)
            fc1_act_fn = gemm_gated if is_glu else gemm_act
            preact, postact = fc1_act_fn(
                x, weight.T, bias=bias, activation=quack_activation, store_preact=True, tuned=False
            )

            ctx.save_for_backward(x, weight, preact)
            ctx.activation_function = activation_function
            ctx.is_glu = is_glu
            ctx.has_bias = bias is not None
            ctx.bias_dtype = bias.dtype if bias is not None else None
            ctx.batch_shape = batch_shape
            ctx.weight_dtype = weight.dtype

            return postact.reshape(*batch_shape, postact.shape[-1])

    @staticmethod
    def backward(ctx, dpostact: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        with torch.amp.autocast("cuda", enabled=False):
            x, weight, preact = ctx.saved_tensors
            dpostact = dpostact.flatten(0, -2).contiguous()

            with torch.enable_grad():
                preact_for_grad = preact.detach().requires_grad_(True)
                postact_for_grad = get_activation_function(ctx.activation_function)(preact_for_grad)

                (dpreact,) = torch.autograd.grad(postact_for_grad, preact_for_grad, dpostact)
                dpreact = dpreact.contiguous()

            dx = None
            if ctx.needs_input_grad[0]:
                dx = gemm(dpreact, weight, tuned=False)
                dx = dx.reshape(*ctx.batch_shape, dx.shape[-1])

            dweight = None
            if ctx.needs_input_grad[1]:
                dweight = gemm(dpreact.T, x, out_dtype=ctx.weight_dtype, tuned=False)

            dbias = None
            if ctx.has_bias and ctx.needs_input_grad[2]:
                dbias = dpreact.sum(0, dtype=ctx.bias_dtype)

            return dx, dweight, dbias, None, None, None


def mlp_fc1_gemm_act(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation_function: str,
) -> torch.Tensor:
    assert not isinstance(weight, DTensor)
    if bias is not None:
        assert not isinstance(bias, DTensor)

    quack_activation = _get_quack_activation(activation_function, _QUACK_GEMM_ACT_MAPPING, "quack_gemm_act")

    return _FusedMLPFC1ActFunc.apply(x, weight, bias, activation_function, quack_activation, False)


def mlp_fc1_gemm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation_function: str,
) -> torch.Tensor:
    assert not isinstance(weight, DTensor)
    if bias is not None:
        assert not isinstance(bias, DTensor)

    quack_activation = _get_quack_activation(activation_function, _QUACK_GEMM_GATED_MAPPING, "quack_gemm_gated")

    return _FusedMLPFC1ActFunc.apply(x, weight, bias, activation_function, quack_activation, True)
