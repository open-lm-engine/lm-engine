# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Any

from ...utils import BaseArgs


class _MLPArgs(BaseArgs):
    mlp_type: str = "MLP"
    intermediate_size: int
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = True

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MLP"


class _MoEArgs(_MLPArgs):
    mlp_type: str = "MoE"
    shared_intermediate_size: int | None = None
    num_experts: int = 8
    num_experts_per_tok: int = 2

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE"
