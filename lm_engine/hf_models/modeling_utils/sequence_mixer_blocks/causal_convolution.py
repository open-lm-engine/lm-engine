# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_causal_conv1d_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import get_activation_function, is_glu
from ..convolution import ParameterizedConv1d
from ..linear import ParameterizedLinear
from ..mlp_blocks.mlp import _get_std_for_linear


if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update


def _apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


def causal_convolution(
    hidden_states: torch.Tensor,
    input_state: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor | None,
    conv1d_num_groups: int,
    return_cache_state: bool,
    activation_string: str,
    conv1d_padding: int,
    conv1d_stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    casual_conv1d_compatible = conv1d_num_groups == conv1d_weight.size(0) and conv1d_weight.size(1) == 1
    sequence_length = hidden_states.size(1)
    kernel_size = conv1d_weight.size(-1)

    assert conv1d_stride == 1
    assert conv1d_padding == kernel_size - 1

    hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)

    if is_kernel_allowed(Kernel.causal_conv1d) and casual_conv1d_compatible:
        use_activation_inside_kernel = activation_string in [None, "silu", "swish"]

        if input_state is None:
            hidden_states = hidden_states.transpose(-1, -2)

            if return_cache_state:
                # F.pad trims the hidden_states if sequence_length > kernel_size
                input_state = F.pad(hidden_states, (kernel_size - sequence_length, 0))

            hidden_states = causal_conv1d_fn(
                x=hidden_states,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=activation_string if use_activation_inside_kernel else None,
            )

            hidden_states = hidden_states.transpose(-1, -2)
        else:
            assert sequence_length == 1

            # we clone to prevent modification in-place
            # torch compile can remove the clone if its not needed
            # this is to prevent silent incorrectness down the line in the model
            input_state_buffer = input_state.clone()
            hidden_states = causal_conv1d_update(
                x=hidden_states,
                conv_state=input_state_buffer,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=activation_string if use_activation_inside_kernel else None,
            )
            input_state = input_state_buffer if return_cache_state else None

        if not use_activation_inside_kernel:
            hidden_states = get_activation_function(activation_string)(hidden_states)
    else:
        if input_state is None:
            hidden_states = hidden_states.transpose(-1, -2)

            if return_cache_state:
                # F.pad trims the hidden_states if sequence_length > kernel_size
                input_state = F.pad(hidden_states, (kernel_size - sequence_length, 0))

            hidden_states = F.conv1d(
                input=hidden_states,
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=conv1d_stride,
                padding=conv1d_padding,
                groups=conv1d_num_groups,
            )

            # removes padding on the right side of the sequence
            hidden_states = hidden_states[..., : -(kernel_size - 1)]
            hidden_states = hidden_states.transpose(-1, -2)
        else:
            assert sequence_length == 1

            input_state = input_state.roll(shifts=-1, dims=-1)
            input_state[..., -1] = hidden_states[:, 0]

            hidden_states = (input_state * conv1d_weight.squeeze(1)).sum(dim=-1)
            if conv1d_bias is not None:
                hidden_states = hidden_states + conv1d_bias

            if not return_cache_state:
                input_state = None

        hidden_states = get_activation_function(activation_string)(hidden_states)
        hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)

    return hidden_states, input_state


class CausalConvolution(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_groups: int,
        activation_function: str,
        add_bias: bool,
        initializer_range: float | None,
        m_width: float,
        init_method: str,
        num_layers: int,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> None:
        super().__init__()

        if use_padding_free_transformer:
            raise NotImplementedError()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.layer_idx = layer_idx
        self.activation_string = activation_function

        std = _get_std_for_linear(initializer_range, init_method, m_width)
        self.input_projection = ParameterizedLinear(hidden_size, in_channels, bias=add_bias, std=std)

        divide_if_divisible(in_channels, num_groups, "")
        divide_if_divisible(out_channels, num_groups, "")

        if is_glu(self.activation_string):
            intermediate_size = divide_if_divisible(out_channels, 2, "")
        else:
            intermediate_size = out_channels

        self.conv1d = ParameterizedConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=add_bias,
            padding=kernel_size - 1,
            groups=num_groups,
            std=std,
        )

        self.activation_function = get_activation_function(self.activation_string)

        self.output_projection = ParameterizedLinear(
            intermediate_size, hidden_size, bias=add_bias, std=std / math.sqrt(2 * num_layers)
        )

        self.casual_conv1d_compatible = self.num_groups == self.in_channels == self.out_channels
        self.use_activation_inside_kernel = self.activation_string in [None, "silu", "swish"]

        mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        mark_parameter_as_mup_learning_rate(self.conv1d.weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)

        mark_parameter_as_no_weight_decay(self.input_projection.bias)
        mark_parameter_as_no_weight_decay(self.conv1d.bias)
        mark_parameter_as_no_weight_decay(self.output_projection.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_state = None if cache_params is None else cache_params.get_cache(self.layer_idx)
        sequence_length = hidden_states.size(1)

        hidden_states = self.input_projection(hidden_states)

        hidden_states, input_state = causal_convolution(
            hidden_states=hidden_states,
            input_state=input_state,
            attention_mask=attention_mask,
            conv1d_weight=self.conv1d.weight,
            conv1d_bias=self.conv1d.bias,
            conv1d_num_groups=self.num_groups,
            return_cache_state=cache_params is not None,
            activation_string=self.activation_string,
            conv1d_padding=self.kernel_size - 1,
            conv1d_stride=1,
        )

        if cache_params is not None:
            cache_params.update(state=input_state, num_tokens_added=sequence_length, layer_idx=self.layer_idx)

        hidden_states = self.output_projection(hidden_states)

        return hidden_states
