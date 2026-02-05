# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Replicate, Shard

from ...dtensors import dtensor_to_tensor, tensor_to_dtensor
from ...utils import ProcessGroupManager, divide_if_divisible
from ..parameter import mark_parameter_as_initialized
from .dtensor_module import DTensorModule
from .TP import get_module_placements


class ParameterizedEmbedding(nn.Embedding, DTensorModule):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        std: float | None = None,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> ParameterizedEmbedding:
        nn.Module.__init__(self)

        self.is_tp_enabled = ProcessGroupManager.is_tensor_parallel_enabled()

        if self.is_tp_enabled:
            self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()
            self.use_padding_free_transformer = use_padding_free_transformer
            self.sequence_parallel = sequence_parallel

            self.vocab_start_index, self.vocab_end_index, num_embeddings_per_tp_rank = get_tensor_parallel_vocab_info(
                num_embeddings
            )

            self.weight = nn.Parameter(
                tensor_to_dtensor(
                    torch.empty(num_embeddings_per_tp_rank, embedding_dim),
                    device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                    current_placement=Shard(0),
                )
            )
        else:
            self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        self.std = std
        self.output_placement = get_module_placements(use_padding_free_transformer, sequence_parallel)

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.is_tp_enabled:
            input = tensor_to_dtensor(input, device_mesh=self.tp_mesh, current_placement=Replicate())

        input = super().forward(input)

        if self.is_tp_enabled:
            input = dtensor_to_tensor(input, device_mesh=self.tp_mesh, desired_placement=self.output_placement)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.std is None:
            super().reset_parameters()
        else:
            self.weight.normal_(mean=0, std=self.std)

        mark_parameter_as_initialized(self.weight)


def get_tensor_parallel_vocab_info(vocab_size: int, make_vocab_size_divisible_by: int = 64) -> tuple[int, int, int]:
    tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
    tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

    divide_if_divisible(make_vocab_size_divisible_by, tp_world_size)

    vocab_size_per_tensor_parallel_rank = (
        make_vocab_size_divisible_by * math.ceil(vocab_size / make_vocab_size_divisible_by)
    ) // tp_world_size

    vocab_start_index = tp_rank * vocab_size_per_tensor_parallel_rank
    vocab_end_index = min((tp_rank + 1) * vocab_size_per_tensor_parallel_rank, vocab_size)

    return vocab_start_index, vocab_end_index, vocab_size_per_tensor_parallel_rank
