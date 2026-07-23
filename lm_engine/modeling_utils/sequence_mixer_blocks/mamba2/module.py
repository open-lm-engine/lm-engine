# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ....enums import Kernel
from ....generation_cache import ConstantCache, GenerationCache, GenerationState
from ....kernels import is_kernel_allowed
from ....parallel import ProcessGroupManager
from ....parameter import (
    mark_parameter_as_initialized,
    mark_parameter_as_mup_learning_rate,
    mark_parameter_as_no_weight_decay,
)
from ....utils import divide_if_divisible
from ...activations import silu
from ...depthwise_causal_convolution import DepthwiseCausalConvolution, _apply_mask_to_padding_states
from ...init_utils import _get_std_for_linear
from ...linear import ParameterizedLinear
from ...normalization import get_normalization_function
from ...softplus_decay_gate import SoftplusDecayGate
from .config import Mamba2Args
from .op import mamba2_cuda, mamba2_torch


class Mamba2(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(
        self,
        hidden_size: int,
        config: Mamba2Args,
        layer_norm_epsilon: float,
        initializer_range: float,
        m_width: float,
        init_method: str,
        num_layers: int,
        layer_idx: int,
        use_depth_scaled_init: bool,
    ) -> Mamba2:
        super().__init__()

        self.num_heads = config.num_heads
        self.hidden_size = hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel_size
        self.intermediate_size = config.intermediate_size
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation_string = config.activation_function

        self.num_groups = config.num_groups
        self.head_dim = divide_if_divisible(config.intermediate_size, config.num_heads, "")
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit

        # 1D convolutional layer
        self.conv_dim = self.intermediate_size + 2 * self.num_groups * self.ssm_state_size
        self.conv1d = DepthwiseCausalConvolution(
            hidden_size=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            activation_function=config.activation_function,
            add_bias=config.use_conv_bias,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=self.conv_kernel_size,
                num_layers=num_layers,
                use_depth_scaled_init=False,
            ),
            use_padding_free_transformer=False,
        )

        # projection of the input hidden states
        self.in_proj = ParameterizedLinear(
            self.hidden_size,
            self.intermediate_size + self.conv_dim + self.num_heads,
            bias=config.add_bias,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=self.hidden_size,
                num_layers=num_layers,
                use_depth_scaled_init=False,
            ),
        )

        self.decay_gate = SoftplusDecayGate(
            hidden_size=None,
            output_size=self.num_heads,
            std=None,
            has_projection=False,
            A_init_min=config.A_init_min,
            A_init_max=config.A_init_max,
            dt_init_min=config.dt_init_min,
            dt_init_max=config.dt_init_max,
            dt_init_floor=config.dt_init_floor,
        )

        self.norm = get_normalization_function(
            config.normalization_function, self.intermediate_size, eps=layer_norm_epsilon
        )

        self.out_proj = ParameterizedLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=config.add_bias,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=self.intermediate_size,
                num_layers=num_layers,
                use_depth_scaled_init=use_depth_scaled_init,
            ),
        )

        self.D = nn.Parameter(torch.empty(self.num_heads))
        mark_parameter_as_no_weight_decay(self.D)

        mark_parameter_as_mup_learning_rate(self.decay_gate.A_log)
        mark_parameter_as_mup_learning_rate(self.D)
        mark_parameter_as_mup_learning_rate(self.conv1d.weight)
        mark_parameter_as_mup_learning_rate(self.in_proj.weight)
        mark_parameter_as_mup_learning_rate(self.out_proj.weight)

        self.reset_parameters()

    def forward(
        self, x: torch.Tensor, cache_params: GenerationCache | None = None, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        S = x.size(1)

        x = _apply_mask_to_padding_states(x, attention_mask)
        x = self.in_proj(x)

        c, h = (
            (None, None)
            if cache_params is None
            else cache_params.get_cache(layer_idx=self.layer_idx, empty_value=(None, None))
        )

        g, x, dt = x.split((self.intermediate_size, self.conv_dim, self.num_heads), dim=-1)
        x, c = self.conv1d(x=x, input_state=c, attention_mask=attention_mask, output_state=cache_params is not None)

        A_neg = -torch.exp(self.decay_gate.A_log.float())
        dt = self.decay_gate.get_dt(x=dt, dt_min=self.time_step_limit[0], dt_max=self.time_step_limit[1])

        groups_time_state_size = self.num_groups * self.ssm_state_size
        x, B, C = x.split((self.intermediate_size, groups_time_state_size, groups_time_state_size), dim=-1)
        B = B.reshape(*B.size()[:-1], self.num_groups, self.ssm_state_size)
        C = C.reshape(*C.size()[:-1], self.num_groups, self.ssm_state_size)

        x, h = (mamba2_cuda if is_kernel_allowed(Kernel.mamba2_ssm) else mamba2_torch)(
            x=x,
            A_neg=A_neg,
            B=B,
            C=C,
            D=self.D,
            dt=dt,
            h=h,
            use_recurrent=S == 1 and h is not None,
            chunk_size=self.chunk_size,
        )

        if cache_params is not None:
            cache_params.update(
                states=(
                    GenerationState(state=c, method=ConstantCache, num_tokens_added=S),
                    GenerationState(state=h, method=ConstantCache, num_tokens_added=S),
                ),
                layer_idx=self.layer_idx,
            )

        x = x * silu(g)
        x = self.norm(x)
        x = self.out_proj(x)

        return x

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.ones_(self.D)
        mark_parameter_as_initialized(self.D)
