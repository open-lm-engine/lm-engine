# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn


class DesyncResidualLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, tensor_parallel_size: int, std: float, bias: bool = True
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.tensor_parallel_size = tensor_parallel_size
        self.std = std

        self.weight = nn.Parameter(torch.empty(self.tensor_parallel_size * self.in_features, self.out_features))

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(self.tensor_parallel_size * self.out_features))

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_tp, batch_size, sequence_length, _ = input.shape

        if input_tp != self.tensor_parallel_size:
            assert input_tp == 1
            input = input.expand(self.tensor_parallel_size, -1, -1, -1)

        # input -> (TP, batch_size, sequence_length, in_features)
        input = input.view(self.tensor_parallel_size, batch_size * sequence_length, -1)
        # input -> (TP, batch_size * sequence_length, in_features)

        weight = self.weight.view(self.tensor_parallel_size, self.in_features, -1)

        if self.bias is None:
            input = torch.bmm(input, weight)
        else:
            bias = self.bias.view(self.tensor_parallel_size, -1).unsqueeze(1)
            input = torch.baddbmm(bias, input, weight, alpha=1, beta=1)

        # input -> (TP, batch_size * sequence_length, out_features)
        input = input.view(self.tensor_parallel_size, batch_size, sequence_length, -1)
        # input -> (TP, batch_size, sequence_length, out_features)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)
        if self.bias is not None:
            self.bias.zero_()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tensor_parallel_size={self.tensor_parallel_size}, bias={self.bias is not None}"
        )
