# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Literal

from ...softplus_decay_gate import SoftPlusDecayGateArgs
from ..mlp import MLPArgs


class DeltaMLPArgs(MLPArgs, SoftPlusDecayGateArgs):
    # Literal is required: mlp_blocks is a discriminated union on "mlp_type"
    # and pydantic rejects plain str discriminator fields.
    mlp_type: Literal["DeltaMLP"] = "DeltaMLP"
    num_ranks: int
    num_heads: int
    use_v_proj: bool
    use_q_l2norm: bool
    use_shortconv: bool
    use_tied_beta: bool
    use_decay_beta: bool
    allow_neg_eigval: bool
    kernel_size: int
    value_scale: float | None = None
    # If True, route b_proj.weight through per-row L2-norm + per-row hyperball in MuonH (each
    # row = one head). If False, b_proj is treated as a single 2D matrix (monolithic NS).
    use_b_proj_per_row_hyperball: bool = False
    use_o_norm: bool = True
    use_k_act: bool = True

    def model_post_init(self, __context: object) -> None:
        assert self.mlp_type == "DeltaMLP"
