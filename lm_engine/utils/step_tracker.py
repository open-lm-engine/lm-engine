# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from .parallel import ProcessGroupManager


_MICRO_BATCH_SIZE: int | None = None
_GRADIENT_ACCUMULATION_STEPS: int | None = None


class StepTracker:
    def __init__(self, micro_batch_size: int, gradient_accumulation_steps: int) -> None:
        global _MICRO_BATCH_SIZE, _GRADIENT_ACCUMULATION_STEPS

        _MICRO_BATCH_SIZE = micro_batch_size
        _GRADIENT_ACCUMULATION_STEPS = gradient_accumulation_steps

    @staticmethod
    def get_local_batch_size() -> int:
        return _MICRO_BATCH_SIZE * _GRADIENT_ACCUMULATION_STEPS

    @staticmethod
    def get_global_batch_size() -> int:
        return StepTracker.get_local_batch_size() * ProcessGroupManager.get_data_parallel_world_size()

    @staticmethod
    def get_gradient_accumulation_steps() -> int:
        return _GRADIENT_ACCUMULATION_STEPS
