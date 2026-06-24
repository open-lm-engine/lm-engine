# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ....accelerator import Accelerator
from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....parameter import mark_parameter_as_mup_learning_rate
from ...activations import get_activation_function, is_glu
from ...dropout import Dropout
from ...init_utils import _get_std_for_linear
from ...linear import ColumnParallelLinear, RowParallelLinear
from ...quack import mlp_fc1_gemm_act, mlp_fc1_gemm_gated
from .config import MLPArgs


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        config: MLPArgs,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        use_depth_scaled_init: bool,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> MLP:
        super().__init__()

        assert isinstance(config, MLPArgs)

        self.is_glu = is_glu(config.activation_function)
        self.activation_function = config.activation_function
        self.accelerator = Accelerator.get_accelerator()

        kwargs = dict(
            in_features=hidden_size,
            out_features=config.intermediate_size,
            bias=config.add_bias,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=hidden_size,
                num_layers=num_layers,
                use_depth_scaled_init=False,
            ),
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        if self.accelerator == Accelerator.tpu and self.is_glu:
            self.up_fc = ColumnParallelLinear(**kwargs)
            self.gate_fc = ColumnParallelLinear(**kwargs)

            mark_parameter_as_mup_learning_rate(self.up_fc.weight)
            mark_parameter_as_mup_learning_rate(self.gate_fc.weight)
        else:
            if self.is_glu:
                kwargs["out_features"] *= 2

            self.c_fc = ColumnParallelLinear(**kwargs)
            mark_parameter_as_mup_learning_rate(self.c_fc.weight)

        self.act = get_activation_function(self.activation_function)

        self.c_proj = RowParallelLinear(
            config.intermediate_size,
            hidden_size,
            bias=config.add_bias,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=config.intermediate_size,
                num_layers=num_layers,
                use_depth_scaled_init=use_depth_scaled_init,
            ),
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        self.dropout = Dropout(
            config.dropout,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._fc1_act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def _fc1_act(self, x: torch.Tensor) -> torch.Tensor:
        if self.accelerator == Accelerator.tpu and self.is_glu:
            u = self.up_fc(x)
            g = self.gate_fc(x)
            return self.act(x=None, u=u, g=g)

        if self.is_glu and is_kernel_allowed(Kernel.quack_gemm_gated):
            return mlp_fc1_gemm_gated(
                x=x,
                weight=self.c_fc.weight,
                bias=self.c_fc.bias,
                activation_function=self.activation_function,
            )

        if not self.is_glu and is_kernel_allowed(Kernel.quack_gemm_act):
            return mlp_fc1_gemm_act(
                x=x,
                weight=self.c_fc.weight,
                bias=self.c_fc.bias,
                activation_function=self.activation_function,
            )

        x = self.c_fc(x)
        x = self.act(x)

        return x
