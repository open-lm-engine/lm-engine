# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate

from ....dtensors import tensor_to_dtensor
from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_causal_conv1d_available, is_mamba_2_ssm_available
from ...cache import GenerationCache
from ...parameter import (
    mark_parameter_as_initialized,
    mark_parameter_as_mup_learning_rate,
    mark_parameter_as_no_weight_decay,
)
from ..activations import get_activation_function
from ..convolution import ParameterizedConv1d
from ..decay_gate import SoftplusDecayGate
from ..linear import ParameterizedLinear
from ..mlp_blocks.mlp import _get_std_for_linear
from ..normalization import get_normalization_function
from .causal_convolution import _apply_mask_to_padding_states


if is_mamba_2_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update


def _pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def _reshape_into_chunks(input_tensor: torch.Tensor, pad_size: int, chunk_size: int) -> torch.Tensor:
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = _pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        input_tensor = input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        input_tensor = input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )

    return input_tensor


def _segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


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
        ssm_state_size: int,
        ssm_intermediate_size: int,
        ssm_num_heads: int,
        conv_kernel_size: int,
        time_step_limit: int,
        add_bias: bool,
        use_conv_bias: bool,
        ssm_activation_function: str,
        num_groups: int,
        chunk_size: int,
        layer_norm_epsilon: float,
        initializer_range: float,
        m_width: float,
        A_init_min: float,
        A_init_max: float,
        dt_init_min: float,
        dt_init_max: float,
        dt_init_floor: float,
        init_method: str,
        normalization_function: str | None,
        num_layers: int,
        layer_idx: int,
    ) -> Mamba2:
        super().__init__()

        self.num_heads = ssm_num_heads
        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = ssm_intermediate_size
        self.layer_idx = layer_idx
        self.use_conv_bias = use_conv_bias

        self.activation_string = ssm_activation_function
        self.activation = get_activation_function(self.activation_string)
        self.use_activation_inside_kernel = self.activation_string in [None, "silu", "swish"]

        self.n_groups = num_groups
        self.head_dim = divide_if_divisible(ssm_intermediate_size, ssm_num_heads, "")
        self.chunk_size = chunk_size

        self.time_step_limit = time_step_limit

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        # 1D convolutional layer
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = ParameterizedConv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            std=std,
        )

        # projection of the input hidden states
        self.in_proj = ParameterizedLinear(
            self.hidden_size,
            self.intermediate_size + self.conv_dim + self.num_heads,
            bias=add_bias,
            std=std,
        )

        self.decay_gate = SoftplusDecayGate(
            hidden_size=None,
            output_size=self.num_heads,
            std=None,
            has_projection=False,
            A_init_min=A_init_min,
            A_init_max=A_init_max,
            dt_init_min=dt_init_min,
            dt_init_max=dt_init_max,
            dt_init_floor=dt_init_floor,
        )

        self.norm = get_normalization_function(normalization_function, self.intermediate_size, eps=layer_norm_epsilon)

        self.D = nn.Parameter(torch.empty(self.num_heads))
        mark_parameter_as_no_weight_decay(self.D)

        self.out_proj = ParameterizedLinear(
            self.intermediate_size, self.hidden_size, bias=add_bias, std=std / math.sqrt(2 * num_layers)
        )

        mark_parameter_as_mup_learning_rate(self.decay_gate.A_log)
        mark_parameter_as_mup_learning_rate(self.D)
        mark_parameter_as_mup_learning_rate(self.conv1d.weight)
        mark_parameter_as_mup_learning_rate(self.in_proj.weight)
        mark_parameter_as_mup_learning_rate(self.out_proj.weight)

        self.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)

        if is_kernel_allowed(Kernel.mamba2_ssm):
            hidden_states = self._cuda_forward(hidden_states, cache_params, attention_mask)
        else:
            hidden_states = self._torch_forward(hidden_states, cache_params, attention_mask)

        return hidden_states

    def _torch_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Gated MLP's linear projection
        hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)
        gate, hidden_states_B_C, dt = projected_states.split(
            [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        conv_state, ssm_state = (None, None) if cache_params is None else cache_params.get_cache(self.layer_idx)
        use_precomputed_states = (
            cache_params is not None and seq_len == 1 and conv_state is not None and ssm_state is not None
        )

        # 2. Convolution sequence transformation
        if use_precomputed_states:
            conv_state = conv_state.roll(shifts=-1, dims=-1)
            conv_state[:, :, -1] = hidden_states_B_C[:, 0, :].to(conv_state.device)
            cache_params.update(conv_state=conv_state, layer_idx=self.layer_idx)

            # We need to guarantee that anything regarding the cache is on the same device
            conv_state = conv_state.to(device=self.conv1d.weight.device)

            hidden_states_B_C = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.activation(hidden_states_B_C)
        else:
            # Init cache
            if cache_params is not None:
                if ssm_state is None:
                    ssm_state = torch.zeros(
                        batch_size,
                        self.num_heads,
                        divide_if_divisible(self.intermediate_size, self.num_heads, ""),
                        self.ssm_state_size,
                        device=projected_states.device,
                        dtype=dtype,
                    )

                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                conv_state = F.pad(
                    hidden_states_B_C_transposed, (self.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0)
                )
                cache_params.update(conv_state=conv_state, ssm_state=ssm_state, layer_idx=self.layer_idx)

            hidden_states_B_C = self.activation(
                self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
            )

        hidden_states_B_C = _apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        # 3. SSM transformation
        A = -torch.exp(self.decay_gate.A_log.float())

        # hidden_states -> B, S, N, head_dim
        # A -> num_heads
        # B -> B, S, ssm_state_size
        # C -> B, S, ssm_state_size
        # dt -> (B, S, N)
        # ssm_state -> (B, N, head_dim, ssm_state_size)

        if use_precomputed_states:
            # We need to guarantee that anything regarding the cache is on the same device
            cache_device = ssm_state.device

            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, 0, :][:, None, ...]
            # dt -> (B, 1, N)
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            # dt -> (B, N, head_dim)
            dt_bias = self.decay_gate.dt_bias[..., None].expand(self.decay_gate.dt_bias.shape[0], self.head_dim)

            dt = F.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            # dt -> (B, N, head_dim)
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            # A -> (N, head_dim, ssm_state_size)
            dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)
            # A -> (N, head_dim, ssm_state_size)
            # dA -> (B, N, head_dim, ssm_state_size)

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            # NOTE: S = 1 actually here
            B, C = [i.reshape(batch_size, self.n_groups, -1)[..., None, :] for i in (B, C)]
            # B, C -> (B, G, 1, ssm_state_size)
            B, C = [
                i.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, i.shape[-1]).contiguous()
                for i in (B, C)
            ]
            # B, C -> (B, G, N / G, ssm_state_size)
            B, C = [i.reshape(batch_size, -1, i.shape[-1]) for i in (B, C)]
            # B, C -> (B, N, ssm_state_size)

            # (B, N, head_dim, 1) * (B, N, 1, ssm_state_size)
            # B is same as k and is shared across heads and dt is used to expand it
            dB = dt[..., None] * B[..., None, :]
            # dB -> (B, N, head_dim, ssm_state_size)

            # Discretize x into dB
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            # hidden_states -> (B, N, head_dim)
            dBx = (dB * hidden_states[..., None]).to(device=cache_device)
            # dBx -> (B, N, head_dim, ssm_state_size)

            # State calculation
            ssm_state = ssm_state * dA + dBx
            cache_params.update(ssm_state=ssm_state, num_tokens_added=seq_len, layer_idx=self.layer_idx)

            ssm_state = ssm_state.to(device=C.device, dtype=C.dtype)  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_state.view(
                batch_size * self.num_heads, self.head_dim, self.ssm_state_size
            )  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            # D skip connection
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # begin ssd naive implementation without einsums
            dt = F.softplus(dt + self.decay_gate.dt_bias)
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
            C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * _pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [
                _reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)
            ]

            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            # This is the analog of a causal mask
            L = torch.exp(_segment_sum(A))

            # Contraction of C and B to get G (attention-weights like)
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)

            # Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            # Compute Y_diag (apply to values)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

            # 2. Compute the state for each intra-chunk
            # (right term of low-rank factorization of off-diagonal blocks; B terms)
            decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
            states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

            # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
            # (middle term of factorization of off-diag blocks; A terms)
            if use_precomputed_states:
                previous_states = cache_params.get_cache(self.layer_idx)[1][:, None, ...].to(device=states.device)
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(_segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
            decay_chunk = decay_chunk.transpose(1, 3)
            new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # 4. Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = torch.exp(A_cumsum)
            C_times_states = C[..., None, :] * states[:, :, None, ...]
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

            # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
            y = Y_diag + Y_off
            # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

            y = y + D_residual
            # Cutting off padded chunks
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)

            # Init cache
            if cache_params is not None:
                cache_params.update(ssm_state=ssm_state, num_tokens_added=seq_len, layer_idx=self.layer_idx)

        scan_output = y * F.silu(gate)
        scan_output = self.norm(scan_output)

        # end ssd naive

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]

        return contextualized_states

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

        conv_state, ssm_state = (None, None) if cache_params is None else cache_params.get_cache(self.layer_idx)
        use_precomputed_states = (
            cache_params is not None and seq_len == 1 and conv_state is not None and ssm_state is not None
        )

        # getting projected states from cache if it exists
        if use_precomputed_states:
            gate, hidden_states_B_C, dt = projected_states.squeeze(1).split(
                [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
            )

            # 2. Convolution sequence transformation
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation_string,
            )
            cache_params.update(conv_state=conv_state, num_tokens_added=seq_len, layer_idx=self.layer_idx)

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
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
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
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = hidden_states * F.silu(gate)
            hidden_states = self.norm(hidden_states)

            # 4. Final linear projection
            out = self.out_proj(hidden_states)[:, None, ...]
        # Fused calculations or step by step if no initialized cache is found
        else:
            A = -torch.exp(self.decay_gate.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

            # 2-4. Fused kernel for conv1d, SSM, and the final projection
            if self.training and cache_params is None:
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

                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)

                # 2. Convolution sequence transformation
                # Init cache
                if cache_params is not None:
                    # storing the states
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    conv_state = F.pad(
                        hidden_states_B_C_transposed,
                        (self.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
                    )
                    cache_params.update(conv_state=conv_state, layer_idx=self.layer_idx)

                hidden_states_B_C = causal_conv1d_fn(
                    x=hidden_states_B_C_transposed,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation_string if self.use_activation_inside_kernel else None,
                ).transpose(1, 2)

                if not self.use_activation_inside_kernel:
                    hidden_states_B_C = self.activation(hidden_states_B_C)

                hidden_states_B_C = _apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )

                # 3. SSM transformation
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
                    **dt_limit_kwargs,
                )

                # Init cache
                if cache_params is not None:
                    cache_params.update(ssm_state=ssm_state, num_tokens_added=seq_len, layer_idx=self.layer_idx)

                scan_output = scan_output.view(batch_size, seq_len, -1)
                # Multiply "gate" branch and apply extra normalization layer
                scan_output = scan_output * F.silu(gate)
                scan_output = self.norm(scan_output)

                # 4. Final linear projection
                out = self.out_proj(scan_output)

        return out

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.ones_(self.D)
        mark_parameter_as_initialized(self.D)
