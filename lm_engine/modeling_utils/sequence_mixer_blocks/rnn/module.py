# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ....enums import Kernel
from ....generation_cache import ConstantCache, GenerationCache, GenerationState
from ....kernels import is_kernel_allowed
from ....parameter import (
    mark_parameter_as_initialized,
    mark_parameter_as_mup_learning_rate,
    mark_parameter_as_no_weight_decay,
)
from ....utils import divide_if_divisible, is_xma_available
from ...activations import get_activation_function, is_glu, silu
from ...depthwise_causal_convolution import DepthwiseCausalConvolution
from ...init_utils import _get_std_for_linear
from ...linear import ParameterizedLinear
from ...normalization import get_normalization_function
from ...sequence_packing import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence
from .config import RNNArgs
from .op import rnn_torch


if is_xma_available():
    from xma import rnn


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: RNNArgs,
        initializer_range: float,
        m_width: float,
        init_method: str,
        num_layers: int,
        layer_idx: int,
        use_depth_scaled_init: bool,
        use_padding_free_transformer: bool,
    ) -> RNN:
        super().__init__()

        self.num_input_heads = config.num_input_heads
        self.num_weight_heads = config.num_weight_heads
        self.num_heads = max(config.num_input_heads, config.num_weight_heads)

        divide_if_divisible(self.num_heads, self.num_input_heads)
        divide_if_divisible(self.num_heads, self.num_weight_heads)

        self.gradient_clipping = config.gradient_clipping

        self.state_head_dim = config.state_head_dim
        self.state_size = self.num_heads * self.state_head_dim
        self.kernel_size = config.kernel_size
        self.activation_string = config.activation_function
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer

        self.x_shape = self.num_input_heads * self.state_head_dim
        self.g_shape = self.num_heads * self.state_head_dim

        self.state_weight_std = _get_std_for_linear(
            initializer_range=initializer_range,
            init_method=init_method,
            m_width=m_width,
            fan_in=self.state_head_dim,
            num_layers=num_layers,
            use_depth_scaled_init=False,
        )

        self.input_projection = ParameterizedLinear(
            input_size,
            self.x_shape + self.g_shape,
            bias=config.add_bias,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=input_size,
                num_layers=num_layers,
                use_depth_scaled_init=False,
            ),
        )

        if self.kernel_size is not None:
            assert not is_glu(self.activation_string)

            self.conv1d = DepthwiseCausalConvolution(
                hidden_size=self.state_size,
                kernel_size=self.kernel_size,
                activation_function=self.activation_string,
                add_bias=config.add_bias,
                std=_get_std_for_linear(
                    initializer_range=initializer_range,
                    init_method=init_method,
                    m_width=m_width,
                    fan_in=self.kernel_size,
                    num_layers=num_layers,
                    use_depth_scaled_init=False,
                ),
                use_padding_free_transformer=use_padding_free_transformer,
            )

            mark_parameter_as_mup_learning_rate(self.conv1d.weight)

        self.activation_function = get_activation_function(self.activation_string)
        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))

        self.output_projection = ParameterizedLinear(
            self.state_size,
            output_size,
            bias=False,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=self.state_size,
                num_layers=num_layers,
                use_depth_scaled_init=use_depth_scaled_init,
            ),
        )

        self.norm = get_normalization_function(config.normalization_function, self.state_size)

        mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        mark_parameter_as_mup_learning_rate(self.state_weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)

        mark_parameter_as_no_weight_decay(self.state_weight)

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if self.use_padding_free_transformer:
            assert cache_params is None
            assert attention_mask is None
        else:
            assert cu_seqlens is None
            assert max_seqlen is None

            B, S = x.size()[:2]

            if attention_mask is not None:
                cu_seqlens, max_seqlen = compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask)
                x = pack_sequence(inputs=x, cu_seqlens=cu_seqlens)

        c, h = (
            (None, None)
            if cache_params is None
            else cache_params.get_cache(layer_idx=self.layer_idx, empty_value=(None, None))
        )

        x = self.input_projection(x)
        x, g = x.split((self.x_shape, self.g_shape), dim=-1)

        if self.kernel_size is None:
            x = self.activation_function(x)
        else:
            x, c = self.conv1d(
                x=x, input_state=c, attention_mask=attention_mask, output_state=cache_params is not None
            )

        x = x.view(*x.size()[:-1], -1, self.state_head_dim)

        if is_kernel_allowed(Kernel.rnn):
            x, h = rnn(
                input=x,
                weight=self.state_weight,
                input_state=h,
                gradient_clipping=self.gradient_clipping,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            assert cu_seqlens is None
            assert max_seqlen is None

            x, h = rnn_torch(x=x, state_weight=self.state_weight, h0=h, gradient_clipping=self.gradient_clipping)

        if not self.use_padding_free_transformer and attention_mask is not None:
            x = unpack_sequence(inputs=x, cu_seqlens=cu_seqlens, output_shape=(B, S, *x.size()[1:]))

        if cache_params is not None:
            cache_params.update(
                states=(
                    GenerationState(state=c, method=ConstantCache, num_tokens_added=S),
                    GenerationState(state=h, method=ConstantCache, num_tokens_added=S),
                ),
                layer_idx=self.layer_idx,
            )

        x = x.flatten(-2, -1)
        x = x * silu(g)
        x = self.norm(x)
        x = self.output_projection(x)

        return x

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.state_weight_std)
        mark_parameter_as_initialized(self.state_weight)

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping}\nweight_shape: {str(self.state_weight.shape)}"
