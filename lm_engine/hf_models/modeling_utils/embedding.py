# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ..parameter import mark_parameter_as_initialized


class ParameterizedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, std: float | None = None) -> ParameterizedEmbedding:
        self.std = std
        super().__init__(num_embeddings, embedding_dim)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            self.weight.normal_(mean=0, std=self.std)

        mark_parameter_as_initialized(self.weight)
