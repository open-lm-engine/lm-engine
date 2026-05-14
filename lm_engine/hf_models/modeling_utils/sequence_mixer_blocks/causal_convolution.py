# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import is_causal_conv1d_available
from ..activations import get_activation_function


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
            hidden_states = hidden_states[..., : 1 - kernel_size]
            hidden_states = hidden_states.transpose(-1, -2)
        else:
            assert sequence_length == 1

            input_state = input_state.roll(shifts=-1, dims=-1)
            input_state[..., -1] = hidden_states[:, 0]

            hidden_states = (input_state * conv1d_weight.squeeze(1)).sum(dim=-1)
            hidden_states = hidden_states[:, None, :]
            if conv1d_bias is not None:
                hidden_states = hidden_states + conv1d_bias

            if not return_cache_state:
                input_state = None

        hidden_states = get_activation_function(activation_string)(hidden_states)
        hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)

    return hidden_states, input_state
