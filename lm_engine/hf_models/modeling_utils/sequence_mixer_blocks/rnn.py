# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_fma_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ...tensor import PackedTensor
from ..linear import ParameterizedLinear
from ..normalization import get_normalization_function


if is_fma_available():
    from fma import KernelBackend
    from fma.modules.rnn import rnn


class RNN(nn.Module):
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
    ) -> RNN:
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.gradient_clipping = gradient_clipping
        self.layer_idx = layer_idx
        self.state_head_dim = divide_if_divisible(self.state_size, self.num_heads, "")

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.state_weight_std = std

        self.input_projection = ParameterizedLinear(self.input_size, 2 * self.state_size, bias=add_bias, std=std)
        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))

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

    def forward(self, x_packed: PackedTensor, cache_params: GenerationCache | None = None) -> PackedTensor:
        cu_seqlens = x_packed.get_cu_seqlens()
        max_seqlen = x_packed.get_max_seqlen()
        x: torch.Tensor = x_packed.get_underlying_tensor(True)

        x = self.input_projection(x)
        x, g = x.chunk(2, dim=-1)
        x = x.view(*x.size()[:-1], self.num_heads, self.state_head_dim)

        if self.scaling_factor != 1:
            x = x * self.scaling_factor

        weight = self.state_weight
        if self.scaling_factor != 1:
            weight = weight * self.scaling_factor

        x = rnn(
            input=x,
            weight=weight,
            input_state=None if cache_params is None else cache_params.get_cache(self.layer_idx),
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.triton if is_kernel_allowed(Kernel.rnn) else KernelBackend.torch,
        )

        if cache_params is not None:
            cache_params.update(
                state=x.get_last_element_along_sequence(),
                num_tokens_added=x.get_cu_seqlens(False),
                layer_idx=self.layer_idx,
            )

        x = x.flatten(-2, -1)
        x = x * F.silu(g)
        x = self.norm(x)
        x = self.output_projection(x)

        x_packed = x_packed.with_new_data(x)

        return x_packed

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping}\nweight_shape: {str(self.state_weight.shape)}"
