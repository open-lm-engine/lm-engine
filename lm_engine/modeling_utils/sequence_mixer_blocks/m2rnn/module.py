# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate

from ....dtensors import tensor_to_dtensor
from ....enums import Kernel
from ....generation_cache import ConstantCache, GenerationCache, GenerationState
from ....kernels import is_kernel_allowed
from ....parallel import ProcessGroupManager
from ....parameter import (
    mark_parameter_as_initialized,
    mark_parameter_as_mup_learning_rate,
    mark_parameter_as_no_weight_decay,
)
from ....utils import divide_if_divisible, is_xma_available
from ...activations import is_glu, silu
from ...depthwise_causal_convolution import DepthwiseCausalConvolution
from ...init_utils import _get_std_for_linear
from ...linear import ParameterizedLinear
from ...normalization import get_normalization_function
from ...rotaters import send_recv
from ...sequence_packing import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence
from ...softplus_decay_gate import SoftplusDecayGate
from .config import M2RNNArgs
from .op import m2rnn_torch


if is_xma_available():
    from xma import m2rnn


class M2RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: M2RNNArgs,
        initializer_range: float,
        m_width: float,
        init_method: str,
        num_layers: int,
        layer_idx: int,
        use_depth_scaled_init: bool,
        use_padding_free_transformer: bool,
    ) -> M2RNN:
        super().__init__()

        self.input_size = input_size
        self.k_head_dim = config.k_head_dim
        self.v_head_dim = config.v_head_dim
        self.output_size = output_size
        self.kernel_size = config.kernel_size
        self.activation_string = config.activation_function
        self.gradient_clipping = config.gradient_clipping
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer
        self.use_residual = config.use_residual

        self.num_q_heads = config.num_q_heads
        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.num_f_heads = config.num_f_heads
        self.num_g_heads = config.num_g_heads
        self.num_weight_heads = config.num_weight_heads

        self.num_heads = max(
            config.num_q_heads, config.num_k_heads, config.num_v_heads, config.num_f_heads, config.num_weight_heads
        )

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
            bias=config.add_bias,
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
            A_init_min=config.A_init_min,
            A_init_max=config.A_init_max,
            dt_init_min=config.dt_init_min,
            dt_init_max=config.dt_init_max,
            dt_init_floor=config.dt_init_floor,
        )

        if config.kernel_size is None:
            assert config.activation_function is None
        else:
            assert self.activation_string is None or not is_glu(self.activation_string)

            self.conv1d = DepthwiseCausalConvolution(
                hidden_size=self.conv_dim,
                kernel_size=config.kernel_size,
                activation_function=self.activation_string,
                add_bias=config.add_bias,
                std=_get_std_for_linear(
                    initializer_range=initializer_range,
                    init_method=init_method,
                    m_width=m_width,
                    fan_in=config.kernel_size,
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

        self.g_norm = get_normalization_function(config.normalization_function, self.num_heads * self.v_head_dim)

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

        f, _ = self.decay_gate(f, final_exponential=True, output_dtype=f.dtype)

        if self.kernel_size is not None:
            x, c = self.conv1d(
                x=x, input_state=c, attention_mask=attention_mask, output_state=cache_params is not None
            )

        q, k, v = x.split((self.q_shape, self.k_shape, self.v_shape), dim=-1)

        q = q.view(*q.size()[:-1], self.num_q_heads, self.k_head_dim)
        k = k.view(*k.size()[:-1], self.num_k_heads, self.k_head_dim)
        v = v.view(*v.size()[:-1], self.num_v_heads, self.v_head_dim)

        if ProcessGroupManager.is_context_parallel_enabled():
            cp_world_size = ProcessGroupManager.get_context_parallel_world_size()
            cp_rank = ProcessGroupManager.get_context_parallel_rank()
        else:
            cp_world_size = 1
            cp_rank = 0

        for i in range(cp_world_size):
            if i == cp_rank:
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
                    assert cu_seqlens is None
                    assert max_seqlen is None

                    x, h = m2rnn_torch(
                        q=q, k=k, v=v, xf=f, W=self.state_weight, h0=h, gradient_clipping=self.gradient_clipping
                    )

                h = send_recv(h)

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
