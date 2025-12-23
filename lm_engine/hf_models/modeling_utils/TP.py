# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from torch.distributed._tensor.placement_types import Placement, Replicate, Shard


def get_module_placements(use_padding_free_transformer: bool, sequence_parallel: bool) -> Placement:
    if sequence_parallel:
        if use_padding_free_transformer:
            placement = Shard(0)
        else:
            placement = Shard(1)
    else:
        placement = Replicate()

    return placement
