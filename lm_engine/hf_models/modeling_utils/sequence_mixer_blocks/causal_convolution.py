# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_causal_conv1d_available
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import get_activation_function
from ..convolution import ParameterizedConv1d
from ..init_utils import _get_std_for_linear


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


class CausalConvolution(nn.Module):
    def __init__(
        self,
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
    ) -> CausalConvolution:
        super().__init__()

        if use_padding_free_transformer:
            raise NotImplementedError()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.layer_idx = layer_idx
        self.activation_string = activation_function

        divide_if_divisible(in_channels, num_groups)
        divide_if_divisible(out_channels, num_groups)

        self.conv1d = ParameterizedConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=add_bias,
            padding=kernel_size - 1,
            groups=num_groups,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=kernel_size,
                num_layers=num_layers,
                use_depth_scaled_init=False,
            ),
        )

        self.activation_function = get_activation_function(self.activation_string)

        self.casual_conv1d_compatible = self.num_groups == self.in_channels == self.out_channels
        self.use_activation_inside_kernel = self.activation_string in [None, "silu", "swish"]

        mark_parameter_as_mup_learning_rate(self.conv1d.weight)
        mark_parameter_as_no_weight_decay(self.conv1d.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_state: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        return_cache_state: bool,
        conv1d_padding: int,
        conv1d_stride: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        W = self.conv1d.weight
        b = self.conv1d.bias

        S = hidden_states.size(1)

        assert conv1d_stride == 1
        assert conv1d_padding == self.kernel_size - 1

        hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)

        if is_kernel_allowed(Kernel.causal_conv1d) and self.casual_conv1d_compatible:
            if input_state is None:
                hidden_states = hidden_states.transpose(-1, -2)

                if return_cache_state:
                    # F.pad trims the hidden_states if sequence_length > kernel_size
                    input_state = F.pad(hidden_states, (self.kernel_size - S, 0))

                hidden_states = causal_conv1d_fn(
                    x=hidden_states,
                    weight=W.squeeze(1),
                    bias=b,
                    activation=self.activation_string if self.use_activation_inside_kernel else None,
                )

                hidden_states = hidden_states.transpose(-1, -2)
            else:
                assert S == 1

                # we clone to prevent modification in-place
                # torch compile can remove the clone if its not needed
                # this is to prevent silent incorrectness down the line in the model
                input_state_buffer = input_state.clone()
                hidden_states = causal_conv1d_update(
                    x=hidden_states,
                    conv_state=input_state_buffer,
                    weight=W.squeeze(1),
                    bias=b,
                    activation=self.activation_string if self.use_activation_inside_kernel else None,
                )
                input_state = input_state_buffer if return_cache_state else None

            if not self.use_activation_inside_kernel:
                hidden_states = self.activation_function(hidden_states)
        else:
            if input_state is None:
                hidden_states = hidden_states.transpose(-1, -2)

                if return_cache_state:
                    # F.pad trims the hidden_states if sequence_length > kernel_size
                    input_state = F.pad(hidden_states, (self.kernel_size - S, 0))

                hidden_states = F.conv1d(
                    input=hidden_states,
                    weight=W,
                    bias=b,
                    stride=conv1d_stride,
                    padding=conv1d_padding,
                    groups=self.num_groups,
                )

                # removes padding on the right side of the sequence
                hidden_states = hidden_states[..., : 1 - self.kernel_size]
                hidden_states = hidden_states.transpose(-1, -2)
            else:
                assert S == 1

                input_state = input_state.roll(shifts=-1, dims=-1)
                input_state[..., -1] = hidden_states[:, 0]

                hidden_states = (input_state * W.squeeze(1)).sum(dim=-1)
                hidden_states = hidden_states[:, None, :]
                if b is not None:
                    hidden_states = hidden_states + b

                if not return_cache_state:
                    input_state = None

            hidden_states = self.activation_function(hidden_states)
            hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)

        return hidden_states, input_state
