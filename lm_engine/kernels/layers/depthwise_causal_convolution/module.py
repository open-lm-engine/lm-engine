# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...accelerator import Accelerator, KernelBackend
from ...utils import is_causal_conv1d_available


if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update


_BASE_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}


def _get_activation_function(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()

    if name not in _BASE_ACTIVATIONS:
        raise ValueError(f"invalid activation function ({name})")

    return _BASE_ACTIVATIONS[name]()


def _apply_mask_to_padding_states(x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = x.dtype
        x = (x * attention_mask[:, :, None]).to(dtype)

    return x


def _get_last_state(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Return the convolution carry as the latest kernel_size raw inputs."""

    # last kernel_size columns of x as passed, not of the original block. Sliced via a positive start index
    # (not x[..., -kernel_size:]) since kernel_size can be 0 (this is always called with self.kernel_size - 1,
    # the input_state/final_state contract used throughout this module), and -0 means "start of tensor", not
    # "empty", to Python's slicing.

    if x.size(-1) < kernel_size:
        return F.pad(x, (kernel_size - x.size(-1), 0))

    return x[..., x.size(-1) - kernel_size :]


class DepthwiseCausalConvolution(nn.Conv1d):
    def __init__(
        self, hidden_size: int, kernel_size: int, activation_function: str | None, add_bias: bool
    ) -> DepthwiseCausalConvolution:
        assert kernel_size > 1

        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=hidden_size,
            bias=add_bias,
        )

        self.activation_string = activation_function
        self.activation_function = _get_activation_function(self.activation_string)
        self.use_activation_inside_kernel = self.activation_string in [None, "silu", "swish"]
        self.kernel_size = kernel_size

    def forward(
        self,
        x: torch.Tensor,
        input_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_state: bool = False,
        *,
        kernel_backend: KernelBackend | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if kernel_backend is None:
            kernel_backend = Accelerator.get_kernel_backend()
        else:
            assert kernel_backend.verify_accelerator()

        BLOCK_SIZE_S = x.size(1)
        S = BLOCK_SIZE_S

        x = _apply_mask_to_padding_states(x, attention_mask)

        final_state = None

        if input_state is None:
            x = x.transpose(-1, -2)

            if output_state:
                final_state = _get_last_state(x, self.kernel_size - 1)

            if kernel_backend == KernelBackend.cuda:
                x = causal_conv1d_fn(
                    x=x,
                    weight=self.weight.squeeze(1),
                    bias=self.bias,
                    initial_states=None,
                    activation=self.activation_string if self.use_activation_inside_kernel else None,
                )

                if not self.use_activation_inside_kernel:
                    x = self.activation_function(x)
            else:
                x = super().forward(x)

                # removes padding on the right side of the sequence
                if self.kernel_size > 1:
                    x = x[..., : 1 - self.kernel_size]

                x = self.activation_function(x)

            x = x.transpose(-1, -2)
        else:
            if S == 1:
                if kernel_backend == KernelBackend.cuda:
                    input_state_buffer = input_state.clone()

                    x = causal_conv1d_update(
                        x=x.squeeze(1),
                        conv_state=input_state_buffer,
                        weight=self.weight.squeeze(1),
                        bias=self.bias,
                        activation=self.activation_string if self.use_activation_inside_kernel else None,
                    )

                    x = x[:, None, :]
                    final_state = input_state_buffer if output_state else None

                    if not self.use_activation_inside_kernel:
                        x = self.activation_function(x)
                else:
                    # input_state: (B, H, K - 1); window: (B, H, K) = input_state's K - 1 history positions
                    # followed by the one new position, i.e. the exact K raw taps this output depends on.
                    window = torch.cat([input_state, x[:, 0][..., None]], dim=-1)

                    x = (window * self.weight.squeeze(1)).sum(dim=-1)
                    x = x[:, None, :]
                    if self.bias is not None:
                        x = x + self.bias

                    # drop the now-oldest position (was input_state[..., 0]) to get back to (B, H, K - 1)
                    final_state = window[..., 1:] if output_state else None

                    x = self.activation_function(x)
            else:
                x = x.transpose(-1, -2)
                # TODO: add fused multi-token continuation support for input_state=[batch, dim, kernel_size - 1]
                # and final_state=[batch, dim, kernel_size - 1]
                x = torch.cat([input_state, x], dim=-1)

                if output_state:
                    final_state = _get_last_state(x, self.kernel_size - 1)

                x = super().forward(x)

                if self.kernel_size > 1:
                    x = x[..., : 1 - self.kernel_size]

                x = x[..., -BLOCK_SIZE_S:]
                x = self.activation_function(x)
                x = x.transpose(-1, -2)

        x = _apply_mask_to_padding_states(x, attention_mask)

        return x, final_state

    def extra_repr(self) -> str:
        output = super().extra_repr()
        return f"{output}\nactivation = {self.activation_string}"
