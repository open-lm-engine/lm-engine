# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...generation_cache import GenerationCache
from ...mixins import Block
from ...modeling_utils import AttentionMaskInfo, PositionInfo


class LadderResidualBlock(Block):
    def forward(
        self,
        current_attention_out: torch.Tensor | None,
        current_mlp_out: torch.Tensor | None,
        residual: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask_info: AttentionMaskInfo = AttentionMaskInfo(),
        position_info: PositionInfo = PositionInfo(),
    ) -> tuple[torch.Tensor]:
        if current_attention_out is not None:
            residual = residual + current_attention_out

        current_attention_out = self.ln_1(residual)
        current_attention_out = self._sequence_mixer_forward(
            current_attention_out,
            cache_params=cache_params,
            attention_mask_info=attention_mask_info,
            position_info=position_info,
        )

        if self.m_residual is not None:
            current_attention_out = current_attention_out * self.m_residual

        if current_mlp_out is not None:
            residual = residual + current_mlp_out

        current_mlp_out = self.ln_2(residual)
        current_mlp_out = self.mlp_block(current_mlp_out)

        if self.m_residual is not None:
            current_mlp_out = current_mlp_out * self.m_residual

        return current_attention_out, current_mlp_out, residual
