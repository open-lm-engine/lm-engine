# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch


class _PrintGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, name: str) -> torch.Tensor:
        ctx.name = name
        return x

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        print(f"gradient for {ctx.name} = {output_grad}")
        return output_grad, None


def print_gradient(x: torch.Tensor, name: str) -> torch.Tensor:
    """print gradient in backward (use only for debugging)

    Args:
        x (torch.Tensor): input tensor
        name (str): additional metadata for the tensor (typically the name of the tensor)

    Returns:
        torch.Tensor: output tensor same as input tensor
    """

    return _PrintGradient.apply(x, name)
