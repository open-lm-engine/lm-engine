# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...enums import Kernel
from ...kernels import is_kernel_allowed
from ...parallel import ProcessGroupManager
from ...parameter import mark_parameter_as_initialized, mark_parameter_as_no_weight_decay
from ...utils import is_causal_conv1d_available
from .activations import get_activation_function
from .rotaters import AllGatherRotater


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


# This prevents torch compile from scheduling preemptively scheduling reduce_scatter
class _ZerosLikeWithBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    @staticmethod
    def backward(ctx, dx: torch.Tensor) -> torch.Tensor:
        return dx * 0


def _zeros_like_with_backward(x: torch.Tensor) -> torch.Tensor:
    return _ZerosLikeWithBackward.apply(x)


def _get_last_state(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Return the convolution carry as the latest kernel_size raw inputs."""

    # last kernel_size columns of x as passed, not of the original block
    if x.size(-1) < kernel_size:
        return F.pad(x, (kernel_size - x.size(-1), 0))

    return x[..., -kernel_size:]


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
        BLOCK_SIZE_S = x.size(1)
        S = BLOCK_SIZE_S

        x = _apply_mask_to_padding_states(x, attention_mask)

        is_cp_enabled = ProcessGroupManager.is_context_parallel_enabled()
        if is_cp_enabled:
            assert input_state is None
            assert not output_state
            assert ProcessGroupManager.get_context_parallel_load_balancing_method() is None
            # because I am lazy and don't want to deal with the other case
            assert BLOCK_SIZE_S >= self.kernel_size - 1

            S *= ProcessGroupManager.get_context_parallel_world_size()

        final_state = None

        if input_state is None:
            if is_cp_enabled and self.kernel_size > 1:
                input_state = x[:, 1 - self.kernel_size :]

                rotater = AllGatherRotater()
                rotater.exchange_buffers(input_state, with_grad=True)

                input_state = rotater.next_buffer()
                if ProcessGroupManager.is_context_parallel_first_rank():
                    input_state = _zeros_like_with_backward(input_state)

            x = x.transpose(-1, -2)

            if output_state:
                final_state = _get_last_state(x, self.kernel_size)

            initial_state_T = input_state.transpose(-1, -2) if input_state is not None else None

            if is_kernel_allowed(Kernel.causal_conv1d):
                x = causal_conv1d_fn(
                    x=x,
                    weight=self.weight.squeeze(1),
                    bias=self.bias,
                    initial_states=initial_state_T,
                    activation=self.activation_string if self.use_activation_inside_kernel else None,
                )

                if not self.use_activation_inside_kernel:
                    x = self.activation_function(x)
            else:
                if is_cp_enabled and self.kernel_size > 1 and initial_state_T is not None:
                    x = torch.cat([initial_state_T, x], dim=-1)

                x = super().forward(x)

                # removes padding on the right side of the sequence
                if self.kernel_size > 1:
                    x = x[..., : 1 - self.kernel_size]

                if is_cp_enabled and self.kernel_size > 1:
                    x = x[..., self.kernel_size - 1 :]

                x = self.activation_function(x)

            x = x.transpose(-1, -2)
        else:
            if S == 1:
                if is_kernel_allowed(Kernel.causal_conv1d):
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
                    final_state = input_state.roll(shifts=-1, dims=-1)
                    final_state[..., -1] = x[:, 0]

                    x = (final_state * self.weight.squeeze(1)).sum(dim=-1)
                    x = x[:, None, :]
                    if self.bias is not None:
                        x = x + self.bias

                    if not output_state:
                        final_state = None

                    x = self.activation_function(x)
            else:
                x = x.transpose(-1, -2)
                # TODO(zhonglin): add fused multi-token continuation support for
                # input_state=[batch, dim, kernel_size] and
                # final_state=[batch, dim, kernel_size].
                x = torch.cat([input_state, x], dim=-1)

                if output_state:
                    final_state = _get_last_state(x, self.kernel_size)

                x = super().forward(x)

                if self.kernel_size > 1:
                    x = x[..., : 1 - self.kernel_size]

                x = x[..., -BLOCK_SIZE_S:]
                x = self.activation_function(x)
                x = x.transpose(-1, -2)

        x = _apply_mask_to_padding_states(x, attention_mask)

        return x, final_state

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
