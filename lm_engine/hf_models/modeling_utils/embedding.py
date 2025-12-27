# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn


class ParameterizedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, std: float | None = None) -> ParameterizedEmbedding:
        self.std = std
        super().__init__(num_embeddings, embedding_dim)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            self.weight.data.normal_(mean=0, std=self.std)
            if self.padding_idx is not None:
                self.weight.data[self.padding_idx].zero_()
