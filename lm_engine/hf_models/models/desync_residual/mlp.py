# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math

import torch
import torch.nn as nn

from ....utils import divide_if_divisible
from ...modeling_utils import get_activation_function, is_glu
from ...modeling_utils.mlp_blocks.mlp import _get_std_for_linear
from .linear import DesyncResidualLinear


class DesyncResidualMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        m_residual: float,
        num_layers: int,
        pretraining_tensor_parallel_size: int,
        all_reduce: bool,
    ) -> None:
        super().__init__()

        self.m_residual = m_residual
        self.tp_world_size = pretraining_tensor_parallel_size
        self.num_layers = num_layers
        self.all_reduce = all_reduce

        intermediate_size = divide_if_divisible(intermediate_size, pretraining_tensor_parallel_size, "")

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.c_fc = DesyncResidualLinear(
            hidden_size,
            2 * intermediate_size if is_glu(activation_function) else intermediate_size,
            tensor_parallel_size=pretraining_tensor_parallel_size,
            bias=add_bias,
            std=std,
        )

        self.act = get_activation_function(activation_function)

        self.c_proj = DesyncResidualLinear(
            intermediate_size,
            hidden_size,
            tensor_parallel_size=pretraining_tensor_parallel_size,
            bias=add_bias,
            std=std / math.sqrt(2 * self.num_layers),
        )

        assert dropout == 0, "residual dropout is not supported with DesyncResidual"

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        if self.all_reduce:
            hidden_states = hidden_states + residual / self.tp_world_size
            hidden_states = hidden_states.sum(dim=0)
        else:
            hidden_states = hidden_states + residual

        return hidden_states
