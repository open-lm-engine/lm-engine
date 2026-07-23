# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ...utils import divide_if_divisible


class _NoLoadBalancer:
    def __init__(self, seq_length: int, world_size: int, device: str | torch.device):
        self.seq_length = seq_length
        self.world_size = world_size
        self.device = device

    def _generate_indices(self, restore: bool = False) -> torch.Tensor:
        seq_length = self.seq_length
        world_size = self.world_size
        chunk_size = divide_if_divisible(seq_length, world_size)
        all_indices = []

        for rank in range(world_size):
            # Generate indices for first chunk of the cp rank
            start = rank * chunk_size
            end = start + chunk_size
            chunk_indices = list(range(start, end))
            # combine the indices for this rank
            all_indices.extend(chunk_indices)

        all_indices_tensor = torch.tensor(all_indices, dtype=torch.int, device=self.device)

        return all_indices_tensor.unsqueeze(0)  # add batch dim


class _HeadTailLoadBalancer(_NoLoadBalancer):
    def _generate_indices(self, restore: bool = False) -> torch.Tensor:
        seq_length = self.seq_length
        world_size = self.world_size
        chunk_size = divide_if_divisible(seq_length, world_size * 2)
        all_indices = []

        for rank in range(world_size):
            # Generate indices for first chunk of the cp rank
            first_chunk_start = rank * chunk_size
            first_chunk_indices = list(range(first_chunk_start, first_chunk_start + chunk_size))

            # Second chunk: positions from the complementary chunk
            second_chunk_idx = world_size * 2 - rank - 1
            second_chunk_start = second_chunk_idx * chunk_size
            second_chunk_indices = list(range(second_chunk_start, second_chunk_start + chunk_size))
            # combine the indices for this rank
            all_indices.extend(first_chunk_indices + second_chunk_indices)

        all_indices_tensor = torch.tensor(all_indices, dtype=torch.long, device=self.device)
        if restore:
            all_indices_tensor = torch.argsort(all_indices_tensor)

        return all_indices_tensor.unsqueeze(0)  # add batch dim
