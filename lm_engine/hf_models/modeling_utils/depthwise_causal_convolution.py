# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...enums import Kernel
from ...kernels import is_kernel_allowed
from ...utils import is_causal_conv1d_available
from ..parameter import mark_parameter_as_initialized, mark_parameter_as_no_weight_decay
from .activations import get_activation_function


if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update


def _apply_mask_to_padding_states(x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = x.dtype
        x = (x * attention_mask[:, :, None]).to(dtype)

    return x


class DepthwiseCausalConvolution(nn.Conv1d):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        activation_function: str,
        add_bias: bool,
        std: float | None,
        use_padding_free_transformer: bool,
    ) -> DepthwiseCausalConvolution:
        if use_padding_free_transformer:
            raise NotImplementedError()

        self.std = std

        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=hidden_size,
            bias=add_bias,
        )

        self.activation_string = activation_function
        self.activation_function = get_activation_function(self.activation_string)
        self.use_activation_inside_kernel = self.activation_string in [None, "silu", "swish"]
        self.kernel_size = kernel_size

        if self.bias is not None:
            mark_parameter_as_no_weight_decay(self.bias)

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        input_state: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        output_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        S = x.size(1)
        x = _apply_mask_to_padding_states(x, attention_mask)

        if is_kernel_allowed(Kernel.causal_conv1d):
            if input_state is None:
                x = x.transpose(-1, -2)

                if output_state:
                    # F.pad trims the x if sequence_length > kernel_size
                    input_state = F.pad(x, (self.kernel_size - S, 0))

                x = causal_conv1d_fn(
                    x=x,
                    weight=self.weight.squeeze(1),
                    bias=self.bias,
                    activation=self.activation_string if self.use_activation_inside_kernel else None,
                )

                x = x.transpose(-1, -2)
            else:
                assert S == 1

                input_state_buffer = input_state.clone()

                x = causal_conv1d_update(
                    x=x.squeeze(1),
                    conv_state=input_state_buffer,
                    weight=self.weight.squeeze(1),
                    bias=self.bias,
                    activation=self.activation_string if self.use_activation_inside_kernel else None,
                )

                x = x[:, None, :]
                input_state = input_state_buffer if output_state else None

            if not self.use_activation_inside_kernel:
                x = self.activation_function(x)
        else:
            if input_state is None:
                x = x.transpose(-1, -2)

                if output_state:
                    # F.pad trims the x if sequence_length > kernel_size
                    input_state = F.pad(x, (self.kernel_size - S, 0))

                x = super().forward(x)

                # removes padding on the right side of the sequence
                if self.kernel_size > 1:
                    x = x[..., : 1 - self.kernel_size]
                x = x.transpose(-1, -2)
            else:
                assert S == 1

                input_state = input_state.roll(shifts=-1, dims=-1)
                input_state[..., -1] = x[:, 0]

                x = (input_state * self.weight.squeeze(1)).sum(dim=-1)
                x = x[:, None, :]
                if self.bias is not None:
                    x = x + self.bias

                if not output_state:
                    input_state = None

            x = self.activation_function(x)
            x = _apply_mask_to_padding_states(x, attention_mask)

        return x, input_state

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            nn.init.normal_(self.weight, mean=0, std=self.std)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias.zero_()

        mark_parameter_as_initialized(self.weight)
        if self.bias is not None:
            mark_parameter_as_initialized(self.bias)
