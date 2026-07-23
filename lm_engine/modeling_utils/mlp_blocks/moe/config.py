# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Any

from ..mlp import MLPArgs


class MoEArgs(MLPArgs):
    mlp_type: str = "MoE"
    shared_intermediate_size: int | None
    num_experts: int
    num_experts_per_tok: int
    shared_expert_gating: bool = False
    normalized_topk: bool = True

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE"
