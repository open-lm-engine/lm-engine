# **************************************************
# Copyright (c) 2026, Mayank Mishra, Jyo Pari, Zhonglin Han
# **************************************************

# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from .....utils import divide_if_divisible, is_fla_available
from ....cache import GenerationCache
from ....parameter import mark_parameter_as_mup_learning_rate
from ...activations import get_activation_function
from ...decay_gate import SoftplusDecayGate
from ...depthwise_causal_convolution import DepthwiseCausalConvolution
from ...init_utils import _get_std_for_linear
from ...linear import LowRankLinear, ParameterizedLinear
from ...normalization import get_normalization_function
from ...sequence_packing import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence


if is_fla_available():
    from .utils import chunk_delta_rule


class DeltaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        use_interleaved_weights: bool,
        num_ranks: int,
        num_heads: int,
        use_v_proj: bool,
        use_q_l2norm: bool,
        use_shortconv: bool,
        use_head_norm: bool,
        use_tied_beta: bool,
        use_decay_beta: bool,
        use_mlp_stream: bool,
        use_input_gate: bool,
        use_output_gate: bool,
        use_output_norm: bool,
        use_zero_init_k: bool,
        allow_neg_eigval: bool,
        conv_size: int,
        layer_idx: int,
        norm_eps: float,
        init_method: str,
        initializer_range: float,
        m_width: float | None,
        A_init_min: float,
        A_init_max: float,
        dt_init_min: float,
        dt_init_max: float,
        dt_init_floor: float,
        num_layers: int,
        use_depth_scaled_init: bool,
        value_scale: float | None,
        use_v_silu: bool,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function
        self.add_bias = add_bias
        self.use_interleaved_weights = use_interleaved_weights
        self.use_padding_free_transformer = use_padding_free_transformer
        self.sequence_parallel = sequence_parallel

        assert not add_bias
        assert dropout == 0
        assert activation_function in ("silu", "swiglu")
        assert not use_interleaved_weights
        assert not use_padding_free_transformer
        assert not sequence_parallel
        assert not (use_input_gate and use_output_gate)

        self.value_scale = 1 / math.sqrt(self.key_dim) if value_scale is None else value_scale
        self.use_v_silu = use_v_silu

        self.use_v_proj = use_v_proj
        self.use_q_l2norm = use_q_l2norm
        self.use_shortconv = use_shortconv
        self.use_head_norm = use_head_norm
        self.use_tied_beta = use_tied_beta
        self.use_decay_beta = use_decay_beta
        self.use_mlp_stream = use_mlp_stream
        self.use_input_gate = use_input_gate
        self.use_output_gate = use_output_gate and use_output_norm
        self.use_output_norm = use_output_norm
        self.conv_size = conv_size
        self.num_ranks = num_ranks
        self.num_heads = num_heads
        self.num_k_heads = self.num_heads
        # just so we keep parameters roughly similar
        self.num_v_heads = 1 if (self.use_input_gate or self.use_output_gate) else 2
        self.num_b_heads = self.num_v_heads if self.use_tied_beta else self.num_heads

        self.k_head_dim = int(self.intermediate_size / self.num_k_heads)
        self.v_head_dim = self.hidden_size

        self.key_dim = int(self.num_k_heads * self.k_head_dim)
        self.value_dim = int(self.num_v_heads * self.v_head_dim)
        self.layer_idx = layer_idx

        assert self.key_dim == self.intermediate_size
        assert self.value_dim == self.hidden_size * self.num_v_heads
        if self.num_v_heads > self.num_k_heads:
            divide_if_divisible(self.num_v_heads, self.num_k_heads)
        else:
            divide_if_divisible(self.num_k_heads, self.num_v_heads)

        up_std = _get_std_for_linear(
            initializer_range=initializer_range,
            init_method=init_method,
            m_width=m_width,
            fan_in=hidden_size,
            num_layers=num_layers,
            use_depth_scaled_init=False,
        )

        kv_size = self.key_dim + self.value_dim
        bg_size = self.num_b_heads + (self.value_dim if self.use_output_gate else 0)
        self.act = get_activation_function("swiglu" if use_input_gate else self.activation_function)
        self.kv_act = get_activation_function(self.activation_function)

        self.q_proj = ParameterizedLinear(
            hidden_size, self.key_dim if not use_input_gate else self.key_dim * 2, bias=False, std=up_std
        )

        num_ranks_std = _get_std_for_linear(
            initializer_range=initializer_range,
            init_method=init_method,
            m_width=m_width,
            fan_in=num_ranks,
            num_layers=num_layers,
            use_depth_scaled_init=False,
        )

        self.k_proj = LowRankLinear(
            hidden_size,
            self.key_dim,
            num_ranks=num_ranks,
            bias=False,
            std_num_ranks=0 if use_zero_init_k else up_std,
            std_high_rank=num_ranks_std,
        )

        if self.use_v_proj:
            self.v_proj = LowRankLinear(
                hidden_size,
                self.value_dim,
                num_ranks=num_ranks,
                bias=False,
                std_num_ranks=up_std,
                std_high_rank=num_ranks_std,
            )
        else:
            assert self.num_v_heads == 1
            assert self.value_dim == self.hidden_size

        self.bg_proj = ParameterizedLinear(hidden_size, bg_size, bias=False, std=up_std)

        if self.use_decay_beta:
            self.decay_gate = SoftplusDecayGate(
                hidden_size=None,
                output_size=self.num_b_heads,
                std=None,
                has_projection=False,
                A_init_min=A_init_min,
                A_init_max=A_init_max,
                dt_init_min=dt_init_min,
                dt_init_max=dt_init_max,
                dt_init_floor=dt_init_floor,
            )

        self.initial_state = ParameterizedLinear(
            self.key_dim,
            self.v_head_dim,
            bias=False,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=self.key_dim,
                num_layers=num_layers,
                use_depth_scaled_init=use_depth_scaled_init,
            ),
        )

        if self.use_shortconv:
            self.kv_conv1d = DepthwiseCausalConvolution(
                hidden_size=kv_size,
                kernel_size=conv_size,
                activation_function=None,
                add_bias=False,
                std=_get_std_for_linear(
                    initializer_range=initializer_range,
                    init_method=init_method,
                    m_width=m_width,
                    fan_in=conv_size,
                    num_layers=num_layers,
                    use_depth_scaled_init=False,
                ),
                use_padding_free_transformer=use_padding_free_transformer,
            )

        if self.use_output_norm:
            self.o_norm = get_normalization_function(
                "rmsnorm",
                # we broadcast the heads
                self.v_head_dim * (self.num_heads if self.use_head_norm else 1),
                eps=norm_eps,
            )

        mark_parameter_as_mup_learning_rate(self.q_proj.weight)
        mark_parameter_as_mup_learning_rate(self.k_proj.u_proj.weight)
        mark_parameter_as_mup_learning_rate(self.k_proj.v_proj.weight)
        if self.use_v_proj:
            mark_parameter_as_mup_learning_rate(self.v_proj.u_proj.weight)
            mark_parameter_as_mup_learning_rate(self.v_proj.v_proj.weight)
        mark_parameter_as_mup_learning_rate(self.bg_proj.weight)
        mark_parameter_as_mup_learning_rate(self.initial_state.weight)
        mark_parameter_as_mup_learning_rate(self.kv_conv1d.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        assert cu_seqlens is None
        assert max_seqlen is None
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else "chunk"
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        if cache_params is None:
            use_cache = False
            conv_state = None
            if self.use_mlp_stream:
                recurrent_state = None
            else:
                recurrent_state = rearrange(
                    self.initial_state.weight,
                    "v (h k) -> 1 h k v",
                    k=self.k_head_dim,
                    v=self.v_head_dim,
                )
        else:
            use_cache = True
            conv_state, recurrent_state = cache_params.get_cache(self.layer_idx)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states) if self.use_v_proj else hidden_states
        bg = self.bg_proj(hidden_states)

        q = self.act(q, is_interleaved=self.use_interleaved_weights) if self.use_input_gate else self.act(q)

        if self.use_mlp_stream:
            o_mlp = self.initial_state(q)

        if self.use_shortconv:
            kv = torch.cat([k, v], dim=-1)
            kv, conv_state = self.kv_conv1d(
                x=kv,
                input_state=conv_state,
                attention_mask=attention_mask,
                output_state=cache_params is not None,
            )

            k, v = kv.split((self.key_dim, self.value_dim), dim=-1)

            if not self.use_v_silu:
                k = self.kv_act(k)
        elif self.use_v_silu:
            kv = self.kv_act(kv)
            k, v = kv.split((self.key_dim, self.value_dim), dim=-1)
        else:
            k, v = kv.split((self.key_dim, self.value_dim), dim=-1)
            k = self.kv_act(k)

        # NOTE this is for an external tracer and not used during training
        if getattr(self, "_capture_kv_post_conv", False):
            self._last_kv_post_conv = torch.cat([k, v], dim=-1).detach()

        v = v * self.value_scale

        if self.use_output_gate:
            b, gate = bg.split((self.num_b_heads, self.value_dim), dim=-1)
        else:
            b = bg

        q = rearrange(q, "... (h d) -> ... h d", d=self.k_head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.k_head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.v_head_dim)
        if self.use_output_gate:
            gate = rearrange(gate, "... (h d) -> ... h d", d=self.v_head_dim)
            if self.num_v_heads < self.num_k_heads:
                gate = repeat(gate, "... h d -> ... (h g) d", g=self.num_k_heads // self.num_v_heads)

        if self.use_decay_beta:
            beta = self.decay_gate(x=b, final_exponential=True, output_dtype=b.dtype)
        else:
            beta = b.sigmoid()

        if self.allow_neg_eigval:
            beta = beta * 2.0

        # NOTE this is for an external tracer and not used during training
        if getattr(self, "_capture_beta", False):
            self._last_beta = beta.detach()

        if attention_mask is not None:
            cu_seqlens, max_seqlen = compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask)
            q, k, v, beta = pack_sequence(inputs=(q, k, v, beta), cu_seqlens=cu_seqlens)

        if mode == "chunk":
            o, recurrent_state = chunk_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache or getattr(self, "_capture_recurrent_state", False),
                cu_seqlens=cu_seqlens,
                use_q_l2norm_in_kernel=self.use_q_l2norm,
                use_k_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # NOTE this is for an external tracer and not used during training
        if getattr(self, "_capture_recurrent_state", False):
            self._last_recurrent_state = recurrent_state.detach()

        if attention_mask is not None:
            o = unpack_sequence(
                inputs=o,
                cu_seqlens=cu_seqlens,
                output_shape=(batch_size, q_len, *hidden_states.size()[1:]),
            )

        if cache_params is not None:
            cache_params.update(
                conv_state=conv_state,
                ssm_state=recurrent_state,
                num_tokens_added=hidden_states.size(1),
                layer_idx=self.layer_idx,
            )

        if self.use_output_gate:
            o = o * F.silu(gate)

        if self.use_output_norm and self.use_head_norm:
            o = rearrange(o, "b t h d -> b t (h d)", h=self.num_heads)
            o = self.o_norm(o)
            o = rearrange(o, "b t (h d) -> b t h d", h=self.num_heads)

        o = reduce(o, "b t h d -> b t d", "sum", h=self.num_heads)
        if self.use_output_norm and not self.use_head_norm:
            o = self.o_norm(o)

        if self.use_mlp_stream:
            o = o + o_mlp

        return o
