# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch.nn as nn

from ...modeling_utils import MLP, get_activation_function, is_glu
from ...modeling_utils.mlp_blocks.mlp import _get_std_for_linear
from ..dropout import Dropout_TP
from ..linear import ColumnParallelLinear, RowParallelLinear


class MLP_TP(MLP):
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
        num_layers: int,
        sequence_parallel: bool = False,
    ) -> MLP_TP:
        nn.Module.__init__(self)

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.c_fc = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size if is_glu(activation_function) else intermediate_size,
            bias=add_bias,
            std=std,
            sequence_parallel=sequence_parallel,
        )

        self.act = get_activation_function(activation_function)

        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=add_bias,
            std=std / math.sqrt(2 * num_layers),
            sequence_parallel=sequence_parallel,
        )

        self.dropout = nn.Identity() if dropout == 0 else Dropout_TP(dropout, sequence_parallel=sequence_parallel)
