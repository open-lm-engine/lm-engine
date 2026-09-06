# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Literal

from ..mlp import MLPArgs


class MoEArgs(MLPArgs):
    mlp_type: Literal["MoE"] = "MoE"
    shared_intermediate_size: int | None
    num_experts: int
    num_experts_per_tok: int
    shared_expert_gating: bool = False
    normalized_topk: bool = True
