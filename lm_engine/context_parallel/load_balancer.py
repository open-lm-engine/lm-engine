# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ..utils import divide_if_divisible


class _HeadTailLoadBalancer:
    def __init__(self, seq_length: int, world_size: int, device: str | torch.device):
        self.seq_length = seq_length
        self.world_size = world_size
        self.device = device

    def _generate_indices(self, restore: bool = False) -> torch.Tensor:
        """
        Generate head-and-tail load balancing indices or restore indices.
        Args:
            restore:
                If True, generate restore indices that map head-and-tail rearranged
                positions back to original positions. If False, generate load
                balance indices that rearrange original positions to head-and-tail pattern.

        Returns:
            The generated indices of shape `(1, seq_len)` because the load-balancing is
            identical within the batch.

        Warning:
            For Multi-Head Attention, we require the masks over the head dimension are identical
            (i.e. the return value of `_generate_indices()` does not have `heads` dimension).

        Example:
            Here is the causal mask for attention where q_len == kv_len == 8:
                            KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 0, 0, 0, 0, 0]
            Q_index [1, 1, 1, 1, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 0, 0]
                    [1, 1, 1, 1, 1, 1, 1, 0]
                    [1, 1, 1, 1, 1, 1, 1, 1]

            Head-tail load-balance strategy rearranges the Q tensor by combining
            Q[0:k] (on seq dim) and Q[-k:] for rank 0, Q[k:2k] and Q[-2k:-k] for
            rank 1, and so on. In python code it looks like:

                k = Q.size(0) // (2 * cp_world_size)
                for rank in range(cp_world_size):
                    reordered_Q[rank * 2 * k : (rank + 1) * 2 * k] = torch.cat(
                        (Q[rank * k : (rank + 1) * k], Q[-(rank + 1) * k : -rank * k])
                    )

            This can also be done by tensor slicing. For the above example, the indices
            tensor for slicing is:
                slice_indices = Tensor([0, 7, 1, 6, 2, 5, 3, 4])

            After reordering QKV using the `slice_indices`, the corresponding mask matrix
            distributing over 2 devices becomes well-balanced:
                            KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 1, 1]
                    [1, 1, 0, 0, 0, 0, 0, 0]    rank 0
                    [1, 1, 1, 1, 1, 1, 1, 0]
            Q_index ------------------------
                    [1, 1, 1, 0, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 0, 0]    rank 1
                    [1, 1, 1, 1, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 0, 0, 0]

            To restore the reordering and putting the tensor back, slicing op can do the
            trick with a `restore_indices` such that:
                slice_indices[restore_indices] == Tensor([0, 1, 2, ...])

            In this way, `reordered_Q[restore_indices]` will just be the original Q.
        """
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

        all_indices_tensor = torch.tensor(all_indices, dtype=torch.int, device=self.device)
        if restore:
            all_indices_tensor = torch.argsort(all_indices_tensor)

        return all_indices_tensor.unsqueeze(0)  # add batch dim
