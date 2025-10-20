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
from ...mask import AttentionMaskInfo
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..linear import ParameterizedLinear
from ..normalization import get_normalization_function


if is_fma_available():
    from fma import KernelBackend
    from fma.layers.gru import gru


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

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.state_weight_std = std

        self.input_projection = ParameterizedLinear(self.input_size, 4 * self.state_size, bias=add_bias, std=std)
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
        self, x: torch.Tensor, attention_mask_info: AttentionMaskInfo, cache_params: GenerationCache | None = None
    ) -> torch.Tensor:
        x = self.input_projection(x)
        x, g = x.split((3 * self.state_size, self.state_size), dim=-1)

        if self.scaling_factor != 1:
            x = x * self.scaling_factor

        x, x_forget, x_reset = x.chunk(3, dim=-1)
        x, x_forget, x_reset = [i.view(T, self.num_heads, self.state_head_dim) for i in (x, x_forget, x_reset)]

        weight = self.state_weight
        if self.scaling_factor != 1:
            weight = weight * self.scaling_factor

        weight, forget_weight, reset_weight = weight.chunk(3, dim=0)

        cu_seqlens = None if attention_mask_info.is_ragged() else attention_mask_info.get_cu_seqlens()
        max_seqlen = None if attention_mask_info.is_ragged() else attention_mask_info.get_max_seqlen()

        x = gru(
            input=attention_mask_info.unpack_sequence(x),
            weight=weight,
            forget_input=x_forget,
            forget_weight=forget_weight,
            reset_input=x_reset,
            reset_weight=reset_weight,
            input_state=None if cache_params is None else cache_params.get_cache(self.layer_idx),
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.triton if is_kernel_allowed(Kernel.gru) else KernelBackend.torch,
        )

        if cache_params is not None:
            if cu_seqlens is None:
                cache_params.update(state=x[:, -1], num_tokens_added=input.size(1), layer_idx=self.layer_idx)
            else:
                cache_params.update(
                    state=x[cu_seqlens[1:] - 1], num_tokens_added=cu_seqlens[1:], layer_idx=self.layer_idx
                )

        x = x.flatten(-2, -1)
        x = x * F.silu(g)
        x = self.norm(x)
        x = self.output_projection(x)

        return x

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping}\nweight_shape: {str(self.state_weight.shape)}"
