# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate

from ....dtensors import tensor_to_dtensor
from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_xma_available
from ...cache import ConstantCache, GenerationCache, GenerationState
from ...parameter import (
    mark_parameter_as_initialized,
    mark_parameter_as_mup_learning_rate,
    mark_parameter_as_no_weight_decay,
)
from ..activations import clip_gradients, is_glu, silu, tanh
from ..convolution import DepthwiseCausalConvolution
from ..decay_gate import SoftplusDecayGate
from ..init_utils import _get_std_for_linear
from ..linear import ParameterizedLinear
from ..normalization import get_normalization_function
from .utils import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence


if is_xma_available():
    from xma import m2rnn


class M2RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        k_head_dim: int,
        v_head_dim: int,
        output_size: int,
        num_q_heads: int,
        num_k_heads: int,
        num_v_heads: int,
        num_f_heads: int,
        num_g_heads: int,
        num_weight_heads: int,
        use_residual: bool,
        kernel_size: int | None,
        activation_function: str | None,
        add_bias: bool,
        gradient_clipping: float | None,
        initializer_range: float,
        m_width: float,
        init_method: str,
        normalization_function: str | None,
        A_init_min: float,
        A_init_max: float,
        dt_init_min: float,
        dt_init_max: float,
        dt_init_floor: float,
        num_layers: int,
        layer_idx: int,
        use_depth_scaled_init: bool,
        use_padding_free_transformer: bool,
    ) -> m2rnn:
        super().__init__()

        self.input_size = input_size
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.activation_string = activation_function
        self.gradient_clipping = gradient_clipping
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer
        self.use_residual = use_residual

        self.num_q_heads = num_q_heads
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.num_f_heads = num_f_heads
        self.num_g_heads = num_g_heads
        self.num_weight_heads = num_weight_heads

        self.num_heads = max(num_q_heads, num_k_heads, num_v_heads, num_f_heads, num_weight_heads)

        divide_if_divisible(self.num_heads, self.num_q_heads)
        divide_if_divisible(self.num_heads, self.num_k_heads)
        divide_if_divisible(self.num_heads, self.num_v_heads)
        divide_if_divisible(self.num_heads, self.num_f_heads)
        divide_if_divisible(self.num_heads, self.num_weight_heads)
        divide_if_divisible(self.num_heads, self.num_g_heads)

        self.q_shape = self.num_q_heads * self.k_head_dim
        self.k_shape = self.num_k_heads * self.k_head_dim
        self.v_shape = self.num_v_heads * self.v_head_dim
        self.g_shape = self.num_g_heads * self.v_head_dim

        self.conv_dim = self.q_shape + self.k_shape + self.v_shape

        self.input_projection = ParameterizedLinear(
            self.input_size,
            self.conv_dim + self.num_f_heads + self.g_shape,
            bias=add_bias,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=self.input_size,
                num_layers=num_layers,
                use_depth_scaled_init=False,
            ),
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

        if kernel_size is None:
            assert activation_function is None
        else:
            assert self.activation_string is None or not is_glu(self.activation_string)

            self.conv1d = DepthwiseCausalConvolution(
                hidden_size=self.conv_dim,
                kernel_size=kernel_size,
                activation_function=self.activation_string,
                add_bias=add_bias,
                std=_get_std_for_linear(
                    initializer_range=initializer_range,
                    init_method=init_method,
                    m_width=m_width,
                    fan_in=kernel_size,
                    num_layers=num_layers,
                    use_depth_scaled_init=False,
                ),
                use_padding_free_transformer=use_padding_free_transformer,
            )

            mark_parameter_as_mup_learning_rate(self.conv1d.weight)

        if self.use_residual:
            self.D = nn.Parameter(torch.empty(self.num_heads, self.v_head_dim))
            mark_parameter_as_no_weight_decay(self.D)

        self.state_weight = nn.Parameter(torch.empty(self.num_weight_heads, self.v_head_dim, self.v_head_dim))
        self.output_projection = ParameterizedLinear(
            self.g_shape,
            self.output_size,
            bias=False,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=self.g_shape,
                num_layers=num_layers,
                use_depth_scaled_init=use_depth_scaled_init,
            ),
        )

        self.g_norm = get_normalization_function(normalization_function, self.num_heads * self.v_head_dim)

        mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        mark_parameter_as_mup_learning_rate(self.state_weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if self.use_padding_free_transformer:
            assert cache_params is None
            assert attention_mask is None
        else:
            assert cu_seqlens is None
            assert max_seqlen is None

            B, S = x.size()[:2]

            if attention_mask is not None:
                cu_seqlens, max_seqlen = compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask)
                x = pack_sequence(inputs=x, cu_seqlens=cu_seqlens)

        c, h = (
            (None, None)
            if cache_params is None
            else cache_params.get_cache(layer_idx=self.layer_idx, empty_value=(None, None))
        )

        x = self.input_projection(x)
        x, f, g = x.split((self.conv_dim, self.num_f_heads, self.g_shape), dim=-1)

        f = self.decay_gate(f, final_exponential=True, output_dtype=f.dtype)

        if self.kernel_size is not None:
            x, c = self.conv1d(
                hidden_states=x,
                input_state=c,
                attention_mask=attention_mask,
                return_cache_state=cache_params is not None,
            )

        q, k, v = x.split((self.q_shape, self.k_shape, self.v_shape), dim=-1)

        q = q.view(*q.size()[:-1], self.num_q_heads, self.k_head_dim)
        k = k.view(*k.size()[:-1], self.num_k_heads, self.k_head_dim)
        v = v.view(*v.size()[:-1], self.num_v_heads, self.v_head_dim)

        if is_kernel_allowed(Kernel.m2rnn):
            x, h = m2rnn(
                query=q,
                key=k,
                value=v,
                weight=self.state_weight,
                forget_input=f,
                input_state=h,
                gradient_clipping=self.gradient_clipping,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            x, h = self._torch_forward(
                q=q,
                k=k,
                v=v,
                xf=f,
                h0=h,
                gradient_clipping=self.gradient_clipping,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        if self.use_residual:
            x = x + v * self.D

        if cache_params is not None:
            cache_params.update(
                states=(
                    GenerationState(state=c, method=ConstantCache, num_tokens_added=S),
                    GenerationState(state=h, method=ConstantCache, num_tokens_added=S),
                ),
                layer_idx=self.layer_idx,
            )

        x = x.flatten(-2, -1)
        g = g.repeat_interleave(self.num_heads // self.num_g_heads, dim=-1)
        x = x * silu(g)
        x = self.g_norm(x)

        x = self.output_projection(x)

        if not self.use_padding_free_transformer and attention_mask is not None:
            x = unpack_sequence(inputs=x, cu_seqlens=cu_seqlens, output_shape=(B, S, *x.size()[1:]))

        return x

    def _torch_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        xf: torch.Tensor,
        h0: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        W = self.state_weight

        Nq = q.size(-2)
        Nk = k.size(-2)
        Nv = v.size(-2)

        Nw = W.size(0)
        Nxf = xf.size(-1)

        N = max(Nq, Nk, Nv, Nw, Nxf)
        V = v.size(-1)

        if cu_seqlens is None:
            B, S, _, K = q.size()
            y = torch.empty(B, S, N, K, V, device=q.device, dtype=q.dtype)
        else:
            B = cu_seqlens.size(0) - 1
            S = max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen
            T, _, K = q.size()

            y = torch.empty(T, N, K, V, device=q.device, dtype=q.dtype)

        if h0 is None:
            h0 = torch.zeros(B, N, K, V, device=k.device, dtype=k.dtype)

        Gq = N // Nq
        Gk = N // Nk
        Gv = N // Nv

        Gw = N // Nw
        Gxf = N // Nxf

        q = q.repeat_interleave(Gq, dim=-2)
        k = k.repeat_interleave(Gk, dim=-2)
        v = v.repeat_interleave(Gv, dim=-2)
        W = W.repeat_interleave(Gw, dim=0)
        xf = xf.repeat_interleave(Gxf, dim=-1)

        # (B, S, N, K, V) = (B, S, N, K, 1) * (B, S, N, 1, V)
        x = k[..., None] * v[..., None, :]
        W = W[None, ...]

        if cu_seqlens is not None:
            h0 = h0.clone()
            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

        for s in range(S):
            if cu_seqlens is None:
                f = xf[:, s, :, None, None]
                # (B, N, K, V) = (B, N, K, V) @ (1, N, V, V) + (B, N, K, V)
                h = h0 @ W + x[:, s]
            else:
                offset = start + s
                unfinished = offset < end
                offset_unfinished = offset[unfinished]

                f = xf[offset_unfinished, :, None, None]
                # (B, N, K, V) = (B, N, K, V) @ (1, N, V, V) + (B, N, K, V)
                h = h0[unfinished] @ W + x[offset_unfinished]

            h = tanh(h)

            if cu_seqlens is None:
                h = f * h0 + (1 - f) * h
            else:
                h = f * h0[unfinished] + (1 - f) * h

            h = clip_gradients(h, gradient_clipping)

            if cu_seqlens is None:
                y[:, s] = h
                h0 = h
            else:
                y[offset_unfinished] = h
                h0[unfinished] = h

        y = q[..., None, :] @ y
        y = y.squeeze(-2)

        return y, h0

    @torch.no_grad()
    def reset_parameters(self) -> None:
        W = torch.eye(self.v_head_dim)
        W = W[None, ...].expand(self.num_heads, -1, -1)

        if isinstance(self.state_weight, DTensor):
            W = tensor_to_dtensor(
                tensor=W,
                device_mesh=self.state_weight.device_mesh,
                current_placement=[Replicate()] * len(self.state_weight.placements),
                desired_placement=self.state_weight.placements,
            )

        self.state_weight.copy_(W)
        mark_parameter_as_initialized(self.state_weight)

        if self.use_residual:
            nn.init.ones_(self.D)
            mark_parameter_as_initialized(self.D)

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping}\nweight_shape: {str(self.state_weight.shape)}"
