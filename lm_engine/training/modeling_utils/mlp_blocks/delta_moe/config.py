# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any, Literal

from ..delta_mlp import DeltaMLPArgs
from ..mlp import MLPArgs


class DeltaMoEArgs(MLPArgs):
    mlp_type: Literal["DeltaMoE"] = "DeltaMoE"
    num_experts: int
    num_experts_per_tok: int
    shared_expert_gating: bool = False
    normalized_topk: bool = True
    delta_mlp: DeltaMLPArgs | None = None

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "DeltaMoE"
