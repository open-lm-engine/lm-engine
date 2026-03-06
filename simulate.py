# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor.placement_types import Replicate

from lm_engine.dtensors import tensor_to_dtensor
from lm_engine.hf_models.modeling_utils.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from lm_engine.utils import ProcessGroupManager


ProcessGroupManager(tensor_parallel_world_size=4)


class M(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qk = ReplicatedLinear(400, 800)
        self.v = ColumnParallelLinear(400, 800)
        self.D = nn.Parameter(tensor_to_dtensor(torch.randn(200)))
        self.o = RowParallelLinear(800, 400)

    def forward(self, x):
        x = tensor_to_dtensor(
            x, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Replicate()
        )
        qk = self.qk(x)
        v = self.v(x)
        q, k = qk.chunk(2, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x + v * self.D

        x = self.o(x)

        return x
