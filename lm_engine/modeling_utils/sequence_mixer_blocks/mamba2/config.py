# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

from ...softplus_decay_gate import SoftPlusDecayGateArgs


class Mamba2Args(SoftPlusDecayGateArgs):
    sequence_mixer_type: str = "mamba2"
    state_size: int
    intermediate_size: int
    num_heads: int
    conv_kernel_size: int
    time_step_limit: tuple[float, float] = (0, float("inf"))
    add_bias: bool = False
    use_conv_bias: bool = True
    activation_function: str
    num_groups: int
    chunk_size: int = 256
    normalization_function: str | None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        assert self.sequence_mixer_type == "mamba2"
