# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed, wait_for_ACT
from ....utils import Accelerator, divide_if_divisible, is_torch_xla_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate
from ..chunk import contiguous_split
from ..dropout import Dropout
from ..dtensor_module import DTensorModule
from ..init_utils import _get_std_for_linear
from ..linear import ColumnParallelLinear, RowParallelLinear
from ..position_embedding import apply_rotary_pos_emb
from .utils import flash_attention


if is_torch_xla_available():
    from torch_xla.experimental.custom_kernel import flash_attention as flash_attention_tpu


def interleave_query_key_value_tensor_for_attention(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    query_heads_per_group = num_heads // num_key_value_heads

    interleaved = []
    for i in range(num_key_value_heads):
        start_index = i * query_heads_per_group * head_dim
        end_index = start_index + query_heads_per_group * head_dim
        interleaved.append(query_weight[start_index:end_index])

        start_index = i * head_dim
        end_index = start_index + head_dim
        interleaved.append(key_weight[start_index:end_index])
        interleaved.append(value_weight[start_index:end_index])

    return torch.cat(interleaved)


def split_query_key_value_tensor_for_attention(
    query_key_value_weight: torch.Tensor, num_heads: int, num_key_value_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_heads_per_group = num_heads // num_key_value_heads
    original_shape = query_key_value_weight.shape

    query_key_value_weight = query_key_value_weight.view(num_key_value_heads, (query_heads_per_group + 2), -1)

    query_weight, key_weight, value_weight = query_key_value_weight.split((query_heads_per_group, 1, 1), 1)

    query_weight = query_weight.reshape(-1, *original_shape[1:])
    key_weight = key_weight.reshape(-1, *original_shape[1:])
    value_weight = value_weight.reshape(-1, *original_shape[1:])

    return query_weight, key_weight, value_weight


class Attention(DTensorModule):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_multiplier: float,
        sliding_window: int | None,
        position_embedding_type: str,
        attention_gate: bool,
        exclusive_self_attention: bool,
        add_bias: bool,
        softmax_dropout: float,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
        use_depth_scaled_init: bool,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> Attention:
        super().__init__()

        self.causal = causal
        self.global_hidden_size = hidden_size
        self.global_num_heads = num_attention_heads
        self.global_num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias
        self.sliding_window = sliding_window
        self.attention_gate = attention_gate
        self.exclusive_self_attention = exclusive_self_attention

        self.use_padding_free_transformer = use_padding_free_transformer
        self.sequence_parallel = sequence_parallel

        divide_if_divisible(self.global_hidden_size, self.global_num_heads)

        self.hidden_size = divide_if_divisible(
            self.global_hidden_size, self.tp_world_size, "hidden_size should be divisible by TP world size"
        )

        self.num_heads = divide_if_divisible(
            self.global_num_heads, self.tp_world_size, "num_heads must be divisible by TP world size"
        )

        self.head_dim = divide_if_divisible(self.hidden_size, self.num_heads, "")
        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        self.num_groups = divide_if_divisible(
            self.global_num_heads,
            self.global_num_key_value_heads,
            f"`num_heads` ({self.global_num_heads}) should be a multiple of `num_key_value_heads` ({self.global_num_key_value_heads})",
        )

        self.num_key_value_heads = divide_if_divisible(
            self.global_num_key_value_heads,
            self.tp_world_size,
            f"`num_key_value_heads` ({self.global_num_key_value_heads}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})",
        )

        self.c_attn = ColumnParallelLinear(
            self.global_hidden_size,
            self.global_hidden_size
            + 2 * self.global_num_key_value_heads * self.head_dim
            + (self.global_hidden_size if self.attention_gate else 0),
            bias=self.add_bias,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=self.global_hidden_size,
                num_layers=num_layers,
                use_depth_scaled_init=False,
            ),
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        self.c_proj = RowParallelLinear(
            self.global_hidden_size,
            self.global_hidden_size,
            bias=self.add_bias,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=self.global_hidden_size,
                num_layers=num_layers,
                use_depth_scaled_init=use_depth_scaled_init,
            ),
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        self.softmax_dropout_p = softmax_dropout

        self.softmax_dropout = Dropout(
            softmax_dropout,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        self.dropout = Dropout(
            dropout,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )

        mark_parameter_as_mup_learning_rate(self.c_attn.weight)
        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        use_flash_attention = (
            is_kernel_allowed(Kernel.flash_attention_2)
            or is_kernel_allowed(Kernel.flash_attention_3)
            or is_kernel_allowed(Kernel.flash_attention_4)
        )

        accelerator = Accelerator.get_accelerator()

        if self.use_padding_free_transformer:
            assert use_flash_attention
            assert past_key_values is None

            T = x.size(0) * (self.tp_world_size if self.sequence_parallel else 1)
            input_shape = (T, self.num_key_value_heads, -1)
            output_shape = (T, -1, self.head_dim)
        else:
            B, S = x.size()[:-1]
            S *= self.tp_world_size if self.sequence_parallel else 1

            input_shape = (B, S, self.num_key_value_heads, -1)
            output_shape = (B, S, -1, self.head_dim)

        x = self.c_attn(x)
        x = x.view(*input_shape)

        if self.attention_gate:
            q, k, v, g = (contiguous_split if Accelerator.get_accelerator() == Accelerator.trainium else torch.split)(
                x,
                (self.num_groups * self.head_dim, self.head_dim, self.head_dim, self.num_groups * self.head_dim),
                dim=-1,
            )

            g = g.reshape(*output_shape)
        else:
            q, k, v = (contiguous_split if Accelerator.get_accelerator() == Accelerator.trainium else torch.split)(
                x, (self.num_groups * self.head_dim, self.head_dim, self.head_dim), dim=-1
            )

        q = q.reshape(*output_shape)

        if self.exclusive_self_attention:
            v_xsa = v
            v_normalized = F.normalize(v, dim=-1)

        if not self.use_padding_free_transformer:
            q, k, v = [i.transpose(1, 2) for i in (q, k, v)]
            if self.attention_gate:
                g = g.transpose(1, 2)

        if self.position_embedding_type == "rope":
            q, k = [apply_rotary_pos_emb(i, cos_sin=rope_cos_sin) for i in (q, k)]

        if past_key_values is not None:
            k, v = past_key_values.update(key_states=k, value_states=v, layer_idx=self.layer_idx)

        if use_flash_attention:
            assert accelerator == Accelerator.cuda

            if not self.use_padding_free_transformer:
                q, k, v = [i.transpose(1, 2) for i in (q, k, v)]
                if self.attention_gate:
                    g = g.transpose(1, 2)

            q, k, v = [wait_for_ACT(i, wait_in_forward=True, wait_in_backward=False) for i in (q, k, v)]

            x = flash_attention(
                q=q,
                k=k,
                v=v,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                attention_mask=attention_mask,
                use_padding_free_transformer=self.use_padding_free_transformer,
                causal=self.causal,
                dropout=self.softmax_dropout_p if self.training else 0,
                softmax_scale=self.attention_multiplier,
                sliding_window=self.sliding_window,
            )

            x = wait_for_ACT(x, wait_in_forward=False, wait_in_backward=True)
        else:
            assert self.sliding_window is None

            if accelerator == Accelerator.tpu:
                assert attention_mask is None
                assert self.softmax_dropout_p == 0

                x = flash_attention_tpu(
                    q,
                    k,
                    v,
                    causal=self.causal if attention_mask is None else False,
                    sm_scale=(
                        1 / math.sqrt(self.head_dim)
                        if self.attention_multiplier is None
                        else self.attention_multiplier
                    ),
                )
            else:
                x = F.scaled_dot_product_attention(
                    query=q,
                    key=k,
                    value=v,
                    attn_mask=attention_mask,
                    dropout_p=self.softmax_dropout_p if self.training else 0,
                    is_causal=self.causal if attention_mask is None else False,
                    scale=self.attention_multiplier,
                    enable_gqa=True,
                )

            x = x.transpose(1, 2)

        if self.exclusive_self_attention:
            x = x - (x * v_xsa).sum(dim=-1, keepdim=True) * v_normalized

        if self.attention_gate:
            x = x * F.sigmoid(g)

        x = x.flatten(-2, -1)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x
