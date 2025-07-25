# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from ...config import CommonConfig


class GPTCrossLayerConfig(CommonConfig):
    model_type = "gpt_crosslayer"

    def __init__(self, sharing_pattern: list[int] | None = None, **kwargs) -> GPTCrossLayerConfig:
        super().__init__(**kwargs)

        if sharing_pattern is None:
            self.sharing_pattern = list(range(self.num_layers))
        else:
            assert all(
                [sharing_pattern[i] == i for i in set(sharing_pattern)]
            ), "a filled sharing pattern doesn't have a parent layer"

            for i in range(len(sharing_pattern) - 1):
                assert sharing_pattern[i] <= sharing_pattern[i + 1]

            for i in range(len(sharing_pattern)):
                assert sharing_pattern[i] >= 0 and sharing_pattern[i] < self.num_layers

            self.sharing_pattern = sharing_pattern

        for i in self.sharing_pattern:
            for j in self.sharing_pattern:
                if i == j:
                    assert self.sequence_mixer_blocks[i] == self.sequence_mixer_blocks[j]
                    assert self.mlp_blocks[i] == self.mlp_blocks[j]

        assert self.init_method == "normal"
