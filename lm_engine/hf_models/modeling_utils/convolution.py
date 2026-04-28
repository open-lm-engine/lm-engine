# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...enums import Kernel
from ...kernels import is_kernel_allowed
from ...utils import divide_if_divisible, is_causal_conv1d_available
from ..parameter import (
    mark_parameter_as_initialized,
    mark_parameter_as_mup_learning_rate,
    mark_parameter_as_no_weight_decay,
)
from .activations import get_activation_function
from .convolution import ParameterizedConv1d


if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update


class ParameterizedConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str | int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        std: float | None = None,
    ) -> ParameterizedConv1d:
        self.std = std
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        mark_parameter_as_no_weight_decay(self.bias)


def _apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


class CausalConvolution(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        activation_function: str,
        add_bias: bool,
        std: float | None,
        use_padding_free_transformer: bool,
    ) -> CausalConvolution:
        if use_padding_free_transformer:
            raise NotImplementedError()

        # _get_std_for_linear(
        #     initializer_range=initializer_range,
        #     init_method=init_method,
        #     m_width=m_width,
        #     fan_in=kernel_size,
        #     num_layers=num_layers,
        #     use_depth_scaled_init=False,
        # )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=groups,
            bias=add_bias,
        )

        self.activation_string = activation_function
        self.activation_function = get_activation_function(self.activation_string)
        self.casual_conv1d_compatible = self.groups == self.in_channels == self.out_channels
        self.use_activation_inside_kernel = self.activation_string in [None, "silu", "swish"]
        self.std = std

        divide_if_divisible(in_channels, groups)
        divide_if_divisible(out_channels, groups)

        mark_parameter_as_mup_learning_rate(self.weight)
        mark_parameter_as_no_weight_decay(self.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_state: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        return_cache_state: bool,
        conv1d_padding: int,
        conv1d_stride: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
                    weight=self.weight.squeeze(1),
                    bias=self.bias,
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
                    weight=self.weight.squeeze(1),
                    bias=self.bias,
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

                hidden_states = super().forward(hidden_states)

                # removes padding on the right side of the sequence
                hidden_states = hidden_states[..., : 1 - self.kernel_size]
                hidden_states = hidden_states.transpose(-1, -2)
            else:
                assert S == 1

                input_state = input_state.roll(shifts=-1, dims=-1)
                input_state[..., -1] = hidden_states[:, 0]

                hidden_states = (input_state * self.weight.squeeze(1)).sum(dim=-1)
                hidden_states = hidden_states[:, None, :]
                if self.bias is not None:
                    hidden_states = hidden_states + self.bias

                if not return_cache_state:
                    input_state = None

            hidden_states = self.activation_function(hidden_states)
            hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)

        return hidden_states, input_state

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
