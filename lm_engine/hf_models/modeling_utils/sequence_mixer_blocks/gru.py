# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_cute_kernels_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..linear import ParameterizedLinear
from ..normalization import get_normalization_function


if is_cute_kernels_available():
    from cute_kernels import KernelBackend
    from cute_kernels.modules.gru import gru_cute


class GRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int,
        num_heads: int,
        add_bias: bool,
        gradient_clipping: float | None,
        initializer_range: float,
        m_width: float,
        init_method: str,
        normalization_function: str | None,
        scaling_factor: float | None,
        num_layers: int,
        layer_idx: int,
    ) -> GRU:
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.gradient_clipping = gradient_clipping
        self.layer_idx = layer_idx
        self.state_head_dim = divide_if_divisible(self.state_size, self.num_heads, "")
        self.is_gated_normalization = normalization_function == "silu_gated_rmsnorm"

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.state_weight_std = std

        self.input_projection = ParameterizedLinear(
            self.input_size,
            3 * self.state_size + (self.state_size if self.is_gated_normalization else 0),
            bias=add_bias,
            std=std,
        )

        self.state_weight = nn.Parameter(torch.empty(3 * self.num_heads, self.state_head_dim, self.state_head_dim))

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.output_projection = ParameterizedLinear(self.state_size, self.output_size, bias=False, std=std)

        self.norm = get_normalization_function(normalization_function, self.state_size)

        self.scaling_factor = scaling_factor
        self.reset_parameters()

        mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        mark_parameter_as_mup_learning_rate(self.state_weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)

        mark_parameter_as_no_weight_decay(self.state_weight)

    def forward(
        self,
        input: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        assert cache_params is None
        assert attention_mask is None

        input_state = None if cache_params is None else cache_params.get_cache(self.layer_idx)

        input = self.input_projection(input)

        if self.is_gated_normalization:
            input, gate = input.split((3 * self.state_size, self.state_size), dim=-1)

        weight = self.state_weight

        if self.scaling_factor != 1:
            input = input * self.scaling_factor
            weight = weight * self.scaling_factor

        input, forget_input, reset_input = input.chunk(3, dim=-1)
        weight, forget_weight, reset_weight = weight.chunk(3, dim=0)

        input, forget_input, reset_input = [
            i.view(*input.size()[:-1], self.num_heads, self.state_head_dim) for i in (input, forget_input, reset_input)
        ]

        input = gru_cute(
            input=input,
            weight=weight,
            forget_input=forget_input,
            forget_weight=forget_weight,
            reset_input=reset_input,
            reset_weight=reset_weight,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.triton if is_kernel_allowed(Kernel.gru_cute) else KernelBackend.torch,
        )

        if cache_params is not None:
            cache_params.update(state=input[:, -1], num_tokens_added=input.size(1), layer_idx=self.layer_idx)

        input = input.view(*input.size()[:-2], -1)

        if self.is_gated_normalization:
            input = self.norm(input, gate)
        else:
            input = self.norm(input)

        input = self.output_projection(input)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping}\nweight_shape: {str(self.state_weight.shape)}"
