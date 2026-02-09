# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh

from ...dtensors import dtensor_to_tensor, tensor_to_dtensor, use_async_tensor_parallel
from ..modeling_utils import ParameterizedEmbedding
from ..modeling_utils.TP import get_module_placements


class LMHead(ParameterizedEmbedding):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compute_with_weight(
            x,
            self.weight,
            use_padding_free_transformer=self.use_padding_free_transformer,
            sequence_parallel=self.sequence_parallel,
            tp_mesh=self.tp_mesh,
        )

    @staticmethod
    def compute_with_weight(
        x: torch.Tensor,
        weight: torch.Tensor,
        use_padding_free_transformer: bool,
        sequence_parallel: bool,
        tp_mesh: DeviceMesh,
    ) -> torch.Tensor:
        function = LMHead._compute_with_weight_compiled if use_async_tensor_parallel() else LMHead._compute_with_weight

        return function(
            input=x,
            weight=weight,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
            tp_mesh=tp_mesh,
        )

    @staticmethod
    def _compute_with_weight(
        x: torch.Tensor,
        weight: torch.Tensor,
        use_padding_free_transformer: bool,
        sequence_parallel: bool,
        tp_mesh: DeviceMesh,
    ) -> torch.Tensor:
        x = tensor_to_dtensor(
            x,
            device_mesh=tp_mesh,
            current_placement=get_module_placements(use_padding_free_transformer, sequence_parallel),
            desired_placement=Replicate(),
        )
        x = F.linear(x, weight)
        x = dtensor_to_tensor(x, device_mesh=tp_mesh, desired_placement=Shard(-1))
        return x

    @torch.compile
    @staticmethod
    def _compute_with_weight_compiled(
        x: torch.Tensor,
        weight: torch.Tensor,
        use_padding_free_transformer: bool,
        sequence_parallel: bool,
        tp_mesh: DeviceMesh,
    ) -> torch.Tensor:
        return LMHead._compute_with_weight(
            input=x,
            weight=weight,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
            tp_mesh=tp_mesh,
        )
