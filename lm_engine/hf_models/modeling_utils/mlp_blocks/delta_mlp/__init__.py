# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange, reduce

from .....parallel import ProcessGroupManager
from .....utils import divide_if_divisible, is_fla_available
from ....cache import ConstantCache, GenerationCache, GenerationState
from ....parameter import (
    mark_parameter_as_initialized,
    mark_parameter_as_mup_learning_rate,
    mark_parameter_as_per_row_hyperball,
)
from ...activations import get_activation_function
from ...decay_gate import SoftplusDecayGate
from ...depthwise_causal_convolution import DepthwiseCausalConvolution
from ...init_utils import _get_std_for_linear
from ...linear import LowRankLinear, ParameterizedLinear
from ...normalization import get_normalization_function
from ...sequence_packing import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence


if is_fla_available():
    from fla.ops.cp import build_cp_context

    from .shortconv import causal_conv1d_cp
    from .utils import chunk_delta_rule, fused_recurrent_delta_rule

    build_cp_context = torch.compiler.disable(build_cp_context)


class DeltaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        num_ranks: int,
        num_heads: int,
        use_v_proj: bool,
        use_q_l2norm: bool,
        use_shortconv: bool,
        use_tied_beta: bool,
        use_decay_beta: bool,
        use_head_o_norm: bool,
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
        use_v_norm: bool,
        use_b_proj_per_row_hyperball: bool = False,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function
        self.add_bias = add_bias
        self.use_padding_free_transformer = use_padding_free_transformer
        self.sequence_parallel = sequence_parallel

        assert not add_bias
        assert dropout == 0
        assert activation_function in ("silu", "swiglu")
        assert not sequence_parallel

        self.use_v_silu = use_v_silu
        self.use_v_proj = use_v_proj
        self.use_v_norm = use_v_norm
        self.use_q_l2norm = use_q_l2norm
        self.use_shortconv = use_shortconv
        self.use_tied_beta = use_tied_beta
        self.use_decay_beta = use_decay_beta
        self.use_head_o_norm = use_head_o_norm
        self.conv_size = conv_size
        self.num_ranks = num_ranks
        self.num_heads = num_heads
        self.num_k_heads = self.num_heads
        # just so we keep parameters roughly similar
        self.num_v_heads = 1
        self.num_b_heads = self.num_v_heads if self.use_tied_beta else self.num_heads

        self.k_head_dim = int(self.intermediate_size / self.num_k_heads)
        self.v_head_dim = self.hidden_size

        self.key_dim = int(self.num_k_heads * self.k_head_dim)
        self.value_dim = int(self.num_v_heads * self.v_head_dim)
        self.value_scale = 1 / math.sqrt(self.key_dim) if value_scale is None else value_scale
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
        self.act = get_activation_function("swiglu")
        self.kv_act = get_activation_function(self.activation_function)

        self.q_proj = ParameterizedLinear(
            hidden_size,
            self.key_dim * 2,
            bias=False,
            std=up_std,
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
            in_features=hidden_size,
            out_features=self.key_dim,
            num_ranks=num_ranks,
            bias=False,
            std_low_rank=up_std,
            std_high_rank=num_ranks_std,
        )

        if self.use_v_proj:
            self.v_proj = LowRankLinear(
                in_features=hidden_size,
                out_features=self.value_dim,
                num_ranks=num_ranks,
                bias=False,
                std_low_rank=up_std,
                std_high_rank=num_ranks_std,
            )
        else:
            assert self.num_v_heads == 1
            assert self.value_dim == self.hidden_size

        self.b_proj = ParameterizedLinear(
            hidden_size,
            self.num_b_heads,
            bias=False,
            std=up_std,
        )

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
            # Dense uses this wrapper directly. CP dispatches its specialized
            # path below; packed inference passes cu_seqlens to the wrapper.
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
                use_padding_free_transformer=False,
            )

        if self.use_head_o_norm:
            self.o_norm = get_normalization_function(
                "rmsnorm",
                self.v_head_dim * self.num_heads,
                eps=norm_eps,
            )
        else:
            self.o_norm = get_normalization_function(
                "rmsnorm",
                self.v_head_dim,
                eps=norm_eps,
            )

        mark_parameter_as_mup_learning_rate(self.q_proj.weight)
        mark_parameter_as_mup_learning_rate(self.k_proj.low_rank_proj.weight)
        mark_parameter_as_mup_learning_rate(self.k_proj.high_rank_proj.weight)
        if self.use_v_proj:
            mark_parameter_as_mup_learning_rate(self.v_proj.low_rank_proj.weight)
            mark_parameter_as_mup_learning_rate(self.v_proj.high_rank_proj.weight)
        mark_parameter_as_mup_learning_rate(self.b_proj.weight)
        # Optional: route b_proj through per-row L2 + per-row hyperball (one head per row),
        # same path as kv_conv1d. Off by default; turned on via use_b_proj_per_row_hyperball.
        if use_b_proj_per_row_hyperball:
            mark_parameter_as_per_row_hyperball(self.b_proj.weight)
        mark_parameter_as_mup_learning_rate(self.initial_state.weight)
        if self.use_shortconv:
            mark_parameter_as_mup_learning_rate(self.kv_conv1d.weight)
            # conv kernel: one row per output channel, each normed + hyperball-projected per row
            mark_parameter_as_per_row_hyperball(self.kv_conv1d.weight)

        self.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """
        Supports three input layouts:
        - dense: hidden_states [B, T, H], cu_seqlens=None.
        - padded: hidden_states [B, T, H] with attention_mask.
        - packed: hidden_states [total_tokens, H] with cu_seqlens.

        Varlen inputs reset convolution state and use one recurrent initial
        state per sequence at cu_seqlens boundaries.
        """
        is_cp_enabled = ProcessGroupManager.is_context_parallel_enabled()

        if self.use_padding_free_transformer:
            assert not self.training
            assert not is_cp_enabled
            assert cache_params is None
            assert attention_mask is None
            assert cu_seqlens is not None
            assert max_seqlen is not None
            assert hidden_states.dim() == 2

            # Packed kernels use physical batch size 1; cu_seqlens carries the
            # logical request batch.
            hidden_states = hidden_states.unsqueeze(0)
            batch_size = cu_seqlens.shape[0] - 1
            q_len = max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen
        else:
            assert cu_seqlens is None
            assert max_seqlen is None

            if attention_mask is not None:
                assert len(attention_mask.shape) == 2, (
                    "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                    "for padding purposes (0 indicating padding). "
                    "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
                )

            batch_size, q_len, _ = hidden_states.shape
            if self.training:
                assert batch_size == 1

        use_fused_recurrent = not self.use_padding_free_transformer and not self.training and q_len <= 64
        mode = "fused_recurrent" if use_fused_recurrent else "chunk"
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        if is_cp_enabled:
            assert mode == "chunk"
            assert batch_size == 1
            assert cache_params is None
            assert attention_mask is None
            assert ProcessGroupManager.get_context_parallel_load_balancing_method() is None

        if cache_params is None:
            use_cache = False
            conv_state = None
            recurrent_state = None
        else:
            use_cache = True
            conv_state, recurrent_state = cache_params.get_cache(layer_idx=self.layer_idx, empty_value=(None, None))

        if recurrent_state is None:
            recurrent_state = rearrange(
                self.initial_state.weight,
                "v (h k) -> 1 h k v",
                k=self.k_head_dim,
                v=self.v_head_dim,
            )

        q = self.q_proj(hidden_states)
        q = self.act(q)

        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states) if self.use_v_proj else hidden_states
        b = self.b_proj(hidden_states)

        if is_cp_enabled:
            cp_world_size = ProcessGroupManager.get_context_parallel_world_size()
            total_seqlen = q_len * cp_world_size
            cu_seqlens_cpu_cp = torch.tensor([0, total_seqlen], dtype=torch.long)
            cp_context = build_cp_context(
                cu_seqlens=cu_seqlens_cpu_cp.to(device=hidden_states.device),
                group=ProcessGroupManager.get_context_parallel_group(),
                conv1d_kernel_size=self.conv_size,
                cu_seqlens_cpu=cu_seqlens_cpu_cp,
            )
        else:
            cp_context = None

        if self.use_shortconv:
            kv = torch.cat([k, v], dim=-1)
            if is_cp_enabled:
                kv = causal_conv1d_cp(
                    x=kv,
                    weight=self.kv_conv1d.weight.squeeze(1),
                    bias=self.kv_conv1d.bias,
                    activation=None,
                    cp_context=cp_context,
                )
            elif self.use_padding_free_transformer:
                kv, conv_state = self.kv_conv1d(
                    x=kv,
                    input_state=conv_state,
                    attention_mask=None,
                    output_state=False,
                    cu_seqlens=cu_seqlens,
                )
            else:
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
            kv = torch.cat([k, v], dim=-1)
            kv = self.kv_act(kv)
            k, v = kv.split((self.key_dim, self.value_dim), dim=-1)
        else:
            k = self.kv_act(k)

        # NOTE this is for an external tracer and not used during training
        if getattr(self, "_capture_kv_post_conv", False):
            self._last_kv_post_conv = torch.cat([k, v], dim=-1).detach()

        q = rearrange(q, "... (h d) -> ... h d", d=self.k_head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.k_head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.v_head_dim)

        v = v * self.value_scale

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
            # pack_sequence returns [total_tokens, ...]; varlen kernels expect
            # physical batch size 1: [1, total_tokens, ...].
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            beta = beta.unsqueeze(0)

        if cu_seqlens is not None:
            num_sequences = cu_seqlens.numel() - 1
            if recurrent_state.size(0) == 1 and num_sequences != 1:
                # TODO(zhonglin): let the varlen delta-rule kernel broadcast
                # shared [1, ...] initial state across logical sequences.
                # Materialize the shared learned state per logical sequence;
                # do not rely on varlen kernels to broadcast [1, ...].
                recurrent_state = recurrent_state.expand(
                    num_sequences,
                    -1,
                    -1,
                    -1,
                ).contiguous()

        output_final_state = use_cache or getattr(self, "_capture_recurrent_state", False)
        if mode == "chunk":
            o, recurrent_state = chunk_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                use_q_l2norm_in_kernel=self.use_q_l2norm,
                use_k_l2norm_in_kernel=True,
                cp_context=cp_context,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=output_final_state,
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
            # unpack_sequence expects packed tokens [total_tokens, ...], while
            # the varlen kernels above return [1, total_tokens, ...].
            o = o.squeeze(0)
            o = unpack_sequence(
                inputs=o,
                cu_seqlens=cu_seqlens,
                output_shape=(batch_size, q_len, *o.size()[1:]),
            )

        if cache_params is not None:
            cache_params.update(
                states=(
                    GenerationState(
                        state=conv_state,
                        method=ConstantCache,
                        num_tokens_added=hidden_states.size(1),
                    ),
                    GenerationState(
                        state=recurrent_state,
                        method=ConstantCache,
                        num_tokens_added=hidden_states.size(1),
                    ),
                ),
                layer_idx=self.layer_idx,
            )

        o = reduce(o, "b t h d -> b t d", "sum", h=self.num_heads)

        if not self.use_head_o_norm:
            o = self.o_norm(o)

        if self.use_padding_free_transformer:
            # Return to the packed input layout.
            o = o.squeeze(0)

        return o

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.use_v_norm:
            # [TODO] this is a footgun if we use efficient initialization
            nn.init.constant_(self.v_norm.weight, val=self.value_scale)
            mark_parameter_as_initialized(self.v_norm.weight)
