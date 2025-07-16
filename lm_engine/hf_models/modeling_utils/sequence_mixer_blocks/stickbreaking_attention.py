# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ....utils import is_stickbreaking_available
from ...cache import GenerationCache
from .softmax_attention import Attention


if is_stickbreaking_available():
    from stickbreaking_attention import sb_attn_varlen


def decoding_stickbreaking(q, k, v, scale=None):
    """
    Stick-breaking attention weights.
    """
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])
    # logits = q @ k[..., :-1, :].transpose(-1, -2) * scale

    assert q.size(2) == 1
    original_dtype = q.dtype
    q = q.float()
    k = k.float()
    logits = q @ k[..., :-1, :].transpose(-1, -2) * scale
    log_z = F.logsigmoid(logits).to(original_dtype)
    log_beta = F.logsigmoid(-logits).to(original_dtype)
    re_cum_log_beta = log_beta.flip(-1).cumsum(dim=-1).flip(-1) - log_beta
    log_att = log_z + re_cum_log_beta
    att: torch.Tensor = log_att.exp()
    v = v[..., :-1, :]
    out = torch.einsum("bhij,bhjd->bhid", att, v)
    return out, 1 - att.sum(dim=-1)


class StickBreakingAttention(Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        assert past_key_values is None
        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        value = value.permute(1, 0, 2)
        hidden_states, rem = sb_attn_varlen(
            q=query.permute(1, 0, 2),
            k=key.permute(1, 0, 2),
            v=value,
            inv_temp=self.attention_multiplier,
            cu_seqlens=cu_seqlens,
            max_seqlens=max_seqlen,
        )
        hidden_states = hidden_states + rem[..., None] * self.head_bias[:, None, :]
        hidden_states = hidden_states.permute(1, 0, 2)

        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.norm(hidden_states)

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    def _prepare_qkv_for_forward_mha(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_q = hidden_states.shape[0]

        hidden_states = hidden_states.view(total_q, self.num_key_value_heads, -1)
        query, key, value = hidden_states.chunk(3, dim=-1)

        return query, key, value

    def _prepare_qkv_for_forward_gqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_q = hidden_states.shape[0]

        hidden_states = hidden_states.view(total_q, self.num_key_value_heads, -1)

        query, key, value = hidden_states.split(
            ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim), dim=-1
        )

        # this needs to be a reshape instead of view sadly
        query = query.reshape(total_q, -1, self.head_dim)
        # key = key.repeat(1, self.num_heads // self.num_key_value_heads, 1)
        # value = value.repeat(1, self.num_heads // self.num_key_value_heads, 1)
        group_size = self.num_heads // self.num_key_value_heads
        key = key.repeat_interleave(repeats=group_size, dim=1)
        value = value.repeat_interleave(repeats=group_size, dim=1)
        return query, key, value

    def _prepare_qkv_for_forward_mqa(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_q = hidden_states.shape[0]

        query, key, value = hidden_states.split((self.hidden_size, self.head_dim, self.head_dim), dim=-1)

        query = query.view(total_q, self.num_heads, -1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        return query, key, value
