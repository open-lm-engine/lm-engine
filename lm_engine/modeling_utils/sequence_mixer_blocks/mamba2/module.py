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
from ....utils import divide_if_divisible, is_mamba_2_ssm_available
from ...activations import silu
from ...depthwise_causal_convolution import DepthwiseCausalConvolution, _apply_mask_to_padding_states
from ...init_utils import _get_std_for_linear
from ...linear import ParameterizedLinear
from ...normalization import get_normalization_function
from ...softplus_decay_gate import SoftplusDecayGate
from .config import Mamba2Args
from .utils import get_cp_initial_ssm_state, mamba2_chunk_scan, mamba2_recurrent_step


if is_mamba_2_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined


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

        self.n_groups = config.num_groups
        self.head_dim = divide_if_divisible(config.intermediate_size, config.num_heads, "")
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit

        # 1D convolutional layer
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
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
        self,
        x: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = _apply_mask_to_padding_states(x, attention_mask)

        if is_kernel_allowed(Kernel.mamba2_ssm):
            x = self._cuda_forward(x, cache_params, attention_mask)
        else:
            x = self._torch_forward(x, cache_params, attention_mask)

        return x

    def _torch_forward(
        self,
        x: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        dtype = x.dtype

        x = _apply_mask_to_padding_states(x, attention_mask)
        x = self.in_proj(x)
        g, x, dt = x.split([self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)

        c, h = (
            (None, None)
            if cache_params is None
            else cache_params.get_cache(layer_idx=self.layer_idx, empty_value=(None, None))
        )

        use_precomputed_states = cache_params is not None and seq_len == 1 and c is not None and h is not None

        if use_precomputed_states:
            x, c = self.conv1d(x=x, input_state=c, attention_mask=attention_mask, output_state=True)
        else:
            if cache_params is not None:
                if h is None:
                    h = torch.zeros(
                        batch_size,
                        self.num_heads,
                        divide_if_divisible(self.intermediate_size, self.num_heads),
                        self.ssm_state_size,
                        device=x.device,
                        dtype=dtype,
                    )

            x, c = self.conv1d(
                x=x, input_state=None, attention_mask=attention_mask, output_state=cache_params is not None
            )

        x = _apply_mask_to_padding_states(x, attention_mask)
        x, B, C = torch.split(
            x,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        A = -torch.exp(self.decay_gate.A_log.float())

        if use_precomputed_states:
            dt = self.decay_gate.get_dt(x=dt[:, 0, :], dt_min=self.time_step_limit[0], dt_max=self.time_step_limit[1])

            x, h = mamba2_recurrent_step(
                hidden_states=x,
                B=B,
                C=C,
                dt=dt,
                A=A,
                D=self.D,
                ssm_state=h,
                num_heads=self.num_heads,
                n_groups=self.n_groups,
                head_dim=self.head_dim,
                ssm_state_size=self.ssm_state_size,
            )

            cache_params.update(
                states=(
                    GenerationState(state=c, method=ConstantCache, num_tokens_added=seq_len),
                    GenerationState(state=h, method=ConstantCache, num_tokens_added=seq_len),
                ),
                layer_idx=self.layer_idx,
            )
        else:
            dt = self.decay_gate.get_dt(x=dt, dt_min=self.time_step_limit[0], dt_max=self.time_step_limit[1])

            x, h = mamba2_chunk_scan(
                hidden_states=x,
                B=B,
                C=C,
                dt=dt,
                A=A,
                D=self.D,
                A_log=self.decay_gate.A_log,
                chunk_size=self.chunk_size,
                num_heads=self.num_heads,
                n_groups=self.n_groups,
                head_dim=self.head_dim,
                ssm_state_size=self.ssm_state_size,
                seq_len=seq_len,
            )

            # Init cache
            if cache_params is not None:
                cache_params.update(
                    states=(
                        GenerationState(state=c, method=ConstantCache, num_tokens_added=seq_len),
                        GenerationState(state=h, method=ConstantCache, num_tokens_added=seq_len),
                    ),
                    layer_idx=self.layer_idx,
                )

        x = x * silu(g)
        x = self.norm(x)
        x = self.out_proj(x.to(dtype))

        return x

    def _cuda_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size

        conv_state, ssm_state = (
            (None, None)
            if cache_params is None
            else cache_params.get_cache(layer_idx=self.layer_idx, empty_value=(None, None))
        )

        use_precomputed_states = (
            cache_params is not None and seq_len == 1 and conv_state is not None and ssm_state is not None
        )

        # getting projected states from cache if it exists
        if use_precomputed_states:
            gate, hidden_states_B_C, dt = projected_states.squeeze(1).split(
                [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
            )

            # 2. Convolution sequence transformation
            hidden_states_B_C, conv_state = self.conv1d(
                x=hidden_states_B_C[:, None, :],
                input_state=conv_state,
                attention_mask=None,
                output_state=cache_params is not None,
            )

            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )

            # 3. SSM transformation
            A = -torch.exp(self.decay_gate.A_log.float())  # (nheads,)
            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.decay_gate.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.reshape(batch_size, self.n_groups, -1)
            C = C.reshape(batch_size, self.n_groups, -1)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(
                ssm_state,
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )

            cache_params.update(
                states=(
                    GenerationState(state=conv_state, method=ConstantCache, num_tokens_added=seq_len),
                    GenerationState(state=ssm_state, method=ConstantCache, num_tokens_added=seq_len),
                ),
                layer_idx=self.layer_idx,
            )

            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = hidden_states * silu(gate)
            hidden_states = self.norm(hidden_states)

            # 4. Final linear projection
            out = self.out_proj(hidden_states)[:, None, ...]
        # Fused calculations or step by step if no initialized cache is found
        else:
            A = -torch.exp(self.decay_gate.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

            # 2-4. Fused kernel for conv1d, SSM, and the final projection
            # The fused kernel does not support passing an initial SSM state, so fall through
            # to the step-by-step path when context parallelism is active.
            if self.training and cache_params is None and not ProcessGroupManager.is_context_parallel_enabled():
                out = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.decay_gate.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,  # was seq_idx
                    activation=self.activation_string,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.eps,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=False,
                    **dt_limit_kwargs,
                )
            else:
                gate, hidden_states_B_C, dt = projected_states.split(
                    [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
                )

                # 2. Convolution sequence transformation
                hidden_states_B_C, conv_state = self.conv1d(
                    x=hidden_states_B_C,
                    input_state=None,
                    attention_mask=attention_mask,
                    output_state=cache_params is not None,
                )

                hidden_states_B_C = _apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )

                # 3. SSM transformation
                if ProcessGroupManager.is_context_parallel_enabled():
                    # Compute the correct initial SSM state for this CP rank.
                    # Pass 1: run scan with zero initial to get the local final state.
                    dt_softplused = self.decay_gate.get_dt(
                        x=dt, dt_min=self.time_step_limit[0], dt_max=self.time_step_limit[1]
                    )

                    scan_output_zero, ssm_state_zero = mamba_chunk_scan_combined(
                        hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                        dt,
                        A,
                        B.view(batch_size, seq_len, self.n_groups, -1),
                        C.view(batch_size, seq_len, self.n_groups, -1),
                        chunk_size=self.chunk_size,
                        D=self.D,
                        z=None,
                        seq_idx=None,
                        return_final_states=True,
                        dt_bias=self.decay_gate.dt_bias,
                        dt_softplus=True,
                        **dt_limit_kwargs,
                    )
                    ssm_state_zero = ssm_state_zero + scan_output_zero.sum().to(ssm_state_zero.dtype) * 0
                    initial_states = get_cp_initial_ssm_state(
                        ssm_state_zero,
                        dt_softplused,
                        self.decay_gate.A_log,
                        self.num_heads,
                        self.head_dim,
                        self.ssm_state_size,
                    )
                else:
                    initial_states = None

                scan_output, ssm_state = mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                    dt,
                    A,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C.view(batch_size, seq_len, self.n_groups, -1),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=self.decay_gate.dt_bias,
                    dt_softplus=True,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )

                # Init cache
                if cache_params is not None:
                    cache_params.update(
                        states=(
                            GenerationState(state=conv_state, method=ConstantCache, num_tokens_added=seq_len),
                            GenerationState(state=ssm_state, method=ConstantCache, num_tokens_added=seq_len),
                        ),
                        layer_idx=self.layer_idx,
                    )

                scan_output = scan_output.view(batch_size, seq_len, -1)
                # Multiply "gate" branch and apply extra normalization layer
                scan_output = scan_output * silu(gate)
                scan_output = self.norm(scan_output)

                # 4. Final linear projection
                out = self.out_proj(scan_output)

        return out

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.ones_(self.D)
        mark_parameter_as_initialized(self.D)
