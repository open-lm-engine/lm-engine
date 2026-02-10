# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...parameter import mark_parameter_as_initialized, mark_parameter_as_no_weight_decay


class ParameterizedLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, std: float | None = None
    ) -> ParameterizedLinear:
        self.std = std
        super().__init__(in_features, out_features, bias)

        mark_parameter_as_no_weight_decay(self.bias)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            nn.init.normal_(self.weight, mean=0, std=self.std)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias.zero_()

        mark_parameter_as_initialized(self.weight)
        mark_parameter_as_initialized(self.bias)
