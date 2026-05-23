# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from ..utils import divide_if_divisible
from .manager import ProcessGroupManager


def get_pipeline_stage_ids_on_current_rank(num_pipeline_stages: int) -> int:
    pp_rank = ProcessGroupManager.get_pipeline_parallel_rank()
    pp_world_size = ProcessGroupManager.get_pipeline_parallel_world_size()

    num_pipeline_stages_per_rank = divide_if_divisible(
        num_pipeline_stages,
        pp_world_size,
        "num_pipeline_stages should be divisible by pipeline_parallel_world_size",
    )

    return tuple(pp_rank + i * pp_world_size for i in range(num_pipeline_stages_per_rank))
