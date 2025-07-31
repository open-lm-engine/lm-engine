# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...utils import BaseArgs


class SamplingParams(BaseArgs):
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None

    def model_post_init(self, __context) -> None:
        if self.top_p is not None:
            assert 0 <= self.top_p <= 1

    def is_greedy(self) -> bool:
        return self.temperature is None or self.temperature == 0
