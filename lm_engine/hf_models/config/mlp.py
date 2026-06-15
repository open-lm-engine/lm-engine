# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ...arguments import BaseArgs
from .sequence_mixer import _SoftPlusDecayArgs


class _MLPArgs(BaseArgs):
    mlp_type: str = "MLP"
    intermediate_size: int
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MLP"


class _MoEArgs(_MLPArgs):
    mlp_type: str = "MoE"
    shared_intermediate_size: int | None = None
    num_experts: int = 8
    num_experts_per_tok: int = 2
    shared_expert_gating: bool = False
    normalized_topk: bool = True

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE"


class _DeltaMLPArgs(_MLPArgs, _SoftPlusDecayArgs):
    mlp_type: str = "DeltaMLP"
    num_ranks: int
    num_heads: int
    use_v_proj: bool
    use_q_l2norm: bool
    use_shortconv: bool
    use_tied_beta: bool
    use_decay_beta: bool
    use_head_o_norm: bool
    allow_neg_eigval: bool
    kernel_size: int
    value_scale: float | None = None
    use_v_silu: bool = True
    use_v_norm: bool = False
    # If True, route b_proj.weight through per-row L2-norm + per-row hyperball in MuonH (each
    # row = one head). If False, b_proj is treated as a single 2D matrix (monolithic NS).
    use_b_proj_per_row_hyperball: bool = False

    def model_post_init(self, __context: object) -> None:
        assert self.mlp_type == "DeltaMLP"
        assert self.A_init_min >= 0
        assert self.A_init_min <= self.A_init_max
        assert self.dt_init_min <= self.dt_init_max
