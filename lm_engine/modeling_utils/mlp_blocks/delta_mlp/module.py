# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from ....generation_cache import ConstantCache, GenerationCache, GenerationState
from ....parallel import ProcessGroupManager
from ....parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_per_row_hyperball
from ....utils import divide_if_divisible, is_fla_available
from ...activations import get_activation_function
from ...depthwise_causal_convolution import DepthwiseCausalConvolution
from ...init_utils import _get_std_for_linear
from ...linear import LowRankLinear, ParameterizedLinear
from ...normalization import get_normalization_function
from ...sequence_packing import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence
from ...softplus_decay_gate import SoftplusDecayGate
from .config import DeltaMLPArgs


_DELTA_MLP_CACHE_NAME = "delta_mlp"


if is_fla_available():
    from fla.ops.cp import build_cp_context

    from .shortconv import causal_conv1d_cp
    from .utils import chunk_delta_rule, fused_recurrent_delta_rule

    build_cp_context = torch.compiler.disable(build_cp_context)


class DeltaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        config: DeltaMLPArgs,
        layer_idx: int,
        norm_eps: float,
        init_method: str,
        initializer_range: float,
        m_width: float | None,
        num_layers: int,
        use_depth_scaled_init: bool,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = config.intermediate_size
        self.activation_function = config.activation_function
        self.use_padding_free_transformer = use_padding_free_transformer

        assert not config.add_bias
        assert config.dropout == 0
        assert not sequence_parallel
        assert not config.allow_neg_eigval
        assert config.activation_function in ("silu", "swiglu")

        self.use_v_proj = config.use_v_proj
        self.use_q_l2norm = config.use_q_l2norm
        self.use_shortconv = config.use_shortconv
        self.use_tied_beta = config.use_tied_beta
        self.use_decay_beta = config.use_decay_beta
        self.use_o_norm = config.use_o_norm
        self.kernel_size = config.kernel_size
        self.num_heads = config.num_heads
        self.num_k_heads = self.num_heads
        # just so we keep parameters roughly similar
        self.num_v_heads = 1
        self.num_b_heads = self.num_v_heads if self.use_tied_beta else self.num_heads

        self.k_head_dim = int(self.intermediate_size / self.num_k_heads)
        self.v_head_dim = self.hidden_size

        self.key_dim = int(self.num_k_heads * self.k_head_dim)
        self.value_dim = int(self.num_v_heads * self.v_head_dim)
        self.value_scale = 1 / math.sqrt(self.key_dim) if config.value_scale is None else config.value_scale
        self.layer_idx = layer_idx

        assert self.key_dim == self.intermediate_size
        assert self.value_dim == self.hidden_size * self.num_v_heads
        if self.num_v_heads > self.num_k_heads:
            divide_if_divisible(self.num_v_heads, self.num_k_heads)
        else:
            divide_if_divisible(self.num_k_heads, self.num_v_heads)

        hidden_std = _get_std_for_linear(
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
            std=hidden_std,
        )

        num_ranks_std = _get_std_for_linear(
            initializer_range=initializer_range,
            init_method=init_method,
            m_width=m_width,
            fan_in=config.num_ranks,
            num_layers=num_layers,
            use_depth_scaled_init=False,
        )

        self.k_proj = LowRankLinear(
            in_features=hidden_size,
            out_features=self.key_dim,
            num_ranks=config.num_ranks,
            bias=False,
            std_low_rank=hidden_std,
            std_high_rank=num_ranks_std,
        )

        if self.use_v_proj:
            self.v_proj = LowRankLinear(
                in_features=hidden_size,
                out_features=self.value_dim,
                num_ranks=config.num_ranks,
                bias=False,
                std_low_rank=hidden_std,
                std_high_rank=num_ranks_std,
            )
        else:
            assert self.num_v_heads == 1
            assert self.value_dim == self.hidden_size

        self.b_proj = ParameterizedLinear(
            hidden_size,
            self.num_b_heads,
            bias=False,
            std=hidden_std,
        )

        if self.use_decay_beta:
            self.decay_gate = SoftplusDecayGate(
                hidden_size=None,
                output_size=self.num_b_heads,
                std=None,
                has_projection=False,
                A_init_min=config.A_init_min,
                A_init_max=config.A_init_max,
                dt_init_min=config.dt_init_min,
                dt_init_max=config.dt_init_max,
                dt_init_floor=config.dt_init_floor,
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
                kernel_size=self.kernel_size,
                activation_function=None,
                add_bias=False,
                std=_get_std_for_linear(
                    initializer_range=initializer_range,
                    init_method=init_method,
                    m_width=m_width,
                    fan_in=self.kernel_size,
                    num_layers=num_layers,
                    use_depth_scaled_init=False,
                ),
                use_padding_free_transformer=False,
            )

        if self.use_o_norm:
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
        if config.use_b_proj_per_row_hyperball:
            mark_parameter_as_per_row_hyperball(self.b_proj.weight)
        mark_parameter_as_mup_learning_rate(self.initial_state.weight)
        if self.use_shortconv:
            mark_parameter_as_mup_learning_rate(self.kv_conv1d.weight)
            # conv kernel: one row per output channel, each normed + hyperball-projected per row
            mark_parameter_as_per_row_hyperball(self.kv_conv1d.weight)

    def initial_recurrent_state(self, batch_size: int = 1) -> torch.Tensor:
        """Return the learned recurrent state in kernel layout [B, H, K, V]."""
        state = rearrange(
            self.initial_state.weight,
            "v (h k) -> 1 h k v",
            k=self.k_head_dim,
            v=self.v_head_dim,
        )
        return state.expand(batch_size, -1, -1, -1).contiguous()

    def _packed_prefill_shortconv(
        self,
        kv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        conv_state: torch.Tensor | None,
        output_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply short-conv over packed prefill segments.

        Prefill uses the fused cu_seqlens path and gathers the final conv
        carry from raw inputs. Cached continuation prepends each segment's
        carry, runs one fused varlen conv, and drops the carry outputs.
        """

        def _gather_last_k_per_segment(kv: torch.Tensor, segment_cu_seqlens: torch.Tensor) -> torch.Tensor:
            starts = segment_cu_seqlens[:-1].to(torch.long)
            ends = segment_cu_seqlens[1:].to(torch.long)
            offsets = torch.arange(self.kv_conv1d.kernel_size, device=kv.device)
            idx = ends[:, None] - self.kv_conv1d.kernel_size + offsets[None, :]
            valid = idx >= starts[:, None]
            gathered = kv[0][idx.clamp_min(0)]
            gathered = gathered * valid[..., None]
            return gathered.transpose(1, 2).contiguous()

        if conv_state is None:
            next_conv_state = _gather_last_k_per_segment(kv, cu_seqlens) if output_state else None
            kv, _ = self.kv_conv1d(
                x=kv,
                input_state=None,
                attention_mask=None,
                output_state=False,
                cu_seqlens=cu_seqlens,
            )
            return kv, next_conv_state

        bounds = cu_seqlens.tolist()
        assert conv_state.size(0) == len(bounds) - 1

        parts = []
        state_tokens = conv_state.transpose(1, 2)
        for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            parts.append(torch.cat([state_tokens[i : i + 1], kv[:, start:end]], dim=1))

        kv = torch.cat(parts, dim=1)
        segment_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        augmented_lengths = segment_lengths + self.kv_conv1d.kernel_size
        augmented_cu_seqlens = torch.zeros_like(cu_seqlens)
        augmented_cu_seqlens[1:] = torch.cumsum(augmented_lengths, dim=0)
        conv_state = _gather_last_k_per_segment(kv, augmented_cu_seqlens) if output_state else None

        kv, _ = self.kv_conv1d(
            x=kv,
            input_state=None,
            attention_mask=None,
            output_state=False,
            cu_seqlens=augmented_cu_seqlens,
        )

        parts = []
        augmented_bounds = augmented_cu_seqlens.tolist()
        state_length = self.kv_conv1d.kernel_size
        for start, end in zip(augmented_bounds[:-1], augmented_bounds[1:]):
            parts.append(kv[:, start + state_length : end])

        return torch.cat(parts, dim=1), conv_state

    def _packed_decode_shortconv(
        self,
        kv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        conv_state: torch.Tensor | None,
        output_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply short-conv over packed unit-length decode segments."""
        assert kv.size(0) == 1
        assert kv.size(1) == cu_seqlens.numel() - 1
        kv = kv.squeeze(0).unsqueeze(1)
        kv, conv_state = self.kv_conv1d(
            x=kv,
            input_state=conv_state,
            attention_mask=None,
            output_state=output_state,
        )
        return kv.squeeze(1).unsqueeze(0), conv_state

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

        Varlen inputs use cu_seqlens boundaries and one recurrent state per
        logical sequence.

        Packed cache contract: conv_state is [num_seq, dim, kernel_size]; a fresh
        sequence is expressed as conv_state=None OR an all-zero state (serving
        uses the latter so fresh and prefixed requests batch together). Note the
        asymmetry — a fresh *conv* state is zero, but a fresh *recurrent* state
        is the learned self.initial_state, not zero (see initial_recurrent_state).
        """
        is_cp_enabled = ProcessGroupManager.is_context_parallel_enabled()

        if self.use_padding_free_transformer:
            assert not self.training
            assert not is_cp_enabled
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

        # Packed decode has one token per active request; packed prefill stays on chunk.
        is_packed_decode = self.use_padding_free_transformer and q_len == 1
        use_fused_recurrent = (
            not self.training and q_len <= 64 and (not self.use_padding_free_transformer or is_packed_decode)
        )
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
            conv_state, recurrent_state = cache_params.get_cache(
                layer_idx=self.layer_idx, empty_value=(None, None), cache_name=_DELTA_MLP_CACHE_NAME
            )

        if recurrent_state is None:
            recurrent_state = self.initial_recurrent_state()

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
                conv1d_kernel_size=self.kernel_size,
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
                kv, conv_state = (
                    self._packed_decode_shortconv if is_packed_decode else self._packed_prefill_shortconv
                )(
                    kv=kv,
                    cu_seqlens=cu_seqlens,
                    conv_state=conv_state,
                    output_state=cache_params is not None,
                )
            else:
                kv, conv_state = self.kv_conv1d(
                    x=kv,
                    input_state=conv_state,
                    attention_mask=attention_mask,
                    output_state=cache_params is not None,
                )

            k, v = kv.split((self.key_dim, self.value_dim), dim=-1)

            k = self.kv_act(k)
        else:
            k = self.kv_act(k)

        q = rearrange(q, "... (h d) -> ... h d", d=self.k_head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.k_head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.v_head_dim)

        v = v * self.value_scale

        if self.use_decay_beta:
            beta = self.decay_gate(x=b, final_exponential=True, output_dtype=b.dtype)
        else:
            beta = b.sigmoid()

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

        output_final_state = use_cache
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

        if attention_mask is not None:
            o = o.squeeze(dim=0)
            # `o` is now head-reduced to (batch, seq, hidden_size)
            o = unpack_sequence(
                inputs=o,
                cu_seqlens=cu_seqlens,
                output_shape=(batch_size, q_len, self.hidden_size),
            )

        if cache_params is not None:
            # Packed serving tracks positions with cu_seqlens and external
            # request state pools, so ConstantCache.get_seq_length() is not a
            # valid scalar for mixed-length requests.
            num_tokens_added = None if self.use_padding_free_transformer else q_len
            cache_params.update(
                states=(
                    GenerationState(state=conv_state, method=ConstantCache, num_tokens_added=num_tokens_added),
                    GenerationState(state=recurrent_state, method=ConstantCache, num_tokens_added=num_tokens_added),
                ),
                layer_idx=self.layer_idx,
                cache_name=_DELTA_MLP_CACHE_NAME,
            )

        if self.use_o_norm:
            o = self.o_norm(o)

        if self.use_padding_free_transformer:
            # Return to the packed input layout.
            o = o.squeeze(0)

        return o
