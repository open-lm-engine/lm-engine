# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ...parameter import mark_parameter_as_mup_learning_rate
from ..activations import get_activation_function, is_glu
from ..dropout import Dropout
from ..linear import ColumnParallelLinear, RowParallelLinear


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        use_interleaved_weights: bool,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> MLP:
        super().__init__()

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.is_glu = is_glu(activation_function)
        self.use_interleaved_weights = use_interleaved_weights

        self.c_fc = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size if self.is_glu else intermediate_size,
            bias=add_bias,
            std=std,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        self.act = get_activation_function(activation_function)

        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=add_bias,
            std=std / math.sqrt(2 * num_layers),
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        self.dropout = Dropout(
            dropout, use_padding_free_transformer=use_padding_free_transformer, sequence_parallel=sequence_parallel
        )

        mark_parameter_as_mup_learning_rate(self.c_fc.weight)
        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x, is_interleaved=self.use_interleaved_weights) if self.is_glu else self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def _get_std_for_linear(initializer_range: float, init_method: str, m_width: float | None) -> float:
    std = initializer_range
    if init_method == "mup":
        std /= math.sqrt(m_width)
    elif init_method != "normal":
        raise ValueError(f"unexpected init_method ({init_method})")

    return std


def interleave_up_gate_tensor_for_mlp(
    up_weight: torch.Tensor, gate_weight: torch.Tensor, is_interleaved: bool
) -> torch.Tensor:
    if is_interleaved:
        W = torch.empty(2 * up_weight.size(0), *up_weight.size()[1:], dtype=up_weight.dtype, device=up_weight.device)
        W[1::2] = up_weight
        W[::2] = gate_weight
    else:
        W = torch.cat([up_weight, gate_weight])

    return W


def split_up_gate_tensor_for_mlp(
    c_fc_weight: torch.Tensor, is_interleaved: bool, dim: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    if is_interleaved:
        if dim == 0:
            u = c_fc_weight[1::2].contiguous()
            g = c_fc_weight[::2].contiguous()
        elif dim == 1:
            u = c_fc_weight[:, 1::2].contiguous()
            g = c_fc_weight[:, ::2].contiguous()
        else:
            raise ValueError
    else:
        u, g = c_fc_weight.chunk(2, dim=dim)

    return u, g
