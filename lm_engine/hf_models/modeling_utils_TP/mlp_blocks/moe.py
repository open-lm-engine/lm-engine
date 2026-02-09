# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard

from ....dtensors import dtensor_to_tensor, tensor_to_dtensor
from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import ProcessGroupManager
from ...loss import add_aux_loss
from ...modeling_utils import Dropout, DTensorModule, MoE, get_activation_function, is_glu
from ...modeling_utils.mlp_blocks.mlp import _get_std_for_linear
from ...modeling_utils.mlp_blocks.moe import (
    ColumnParallelExperts,
    ReplicatedLinear_TP,
    RowParallelExperts,
    SharedExpertsColumnParallelLinear,
    SharedExpertsRowParallelLinear,
)


class MoE_TP(MoE, DTensorModule):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        use_interleaved_weights: bool,
        shared_expert_gating: bool,
        normalized_topk: bool,
        num_experts: int,
        num_experts_per_tok: int,
        activation_function: str,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        use_padding_free_transformer: bool,
        sequence_parallel: bool = False,
    ) -> MoE_TP:
        nn.Module.__init__(self)

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.use_padding_free_transformer = use_padding_free_transformer
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.shared_expert_gating = shared_expert_gating
        self.normalized_topk = normalized_topk
        self.use_interleaved_weights = use_interleaved_weights

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

        self.gate = ReplicatedLinear_TP(
            in_features=self.hidden_size,
            out_features=num_experts,
            bias=False,
            std=std,
        )

        self.c_fc = ColumnParallelExperts(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu(activation_function) else self.intermediate_size,
            std=std,
        )

        if self.shared_intermediate_size is not None:
            self.c_fc_shared = SharedExpertsColumnParallelLinear(
                in_features=self.hidden_size,
                out_features=(
                    2 * self.shared_intermediate_size if is_glu(activation_function) else self.shared_intermediate_size
                ),
                bias=False,
                std=std,
            )

        self.act = get_activation_function(activation_function)

        std /= math.sqrt(2 * num_layers)

        self.c_proj = RowParallelExperts(
            num_experts=num_experts, in_features=self.intermediate_size, out_features=self.hidden_size, std=std
        )
        if self.shared_intermediate_size is not None:
            self.c_proj_shared = SharedExpertsRowParallelLinear(
                in_features=self.shared_intermediate_size,
                out_features=self.hidden_size,
                bias=False,
                std=std,
            )

        self.dropout = Dropout(dropout)
        self.placement = Shard(0) if sequence_parallel else Replicate()

        self.is_hopper_or_newer_gpu = torch.cuda.is_available() and torch.cuda.get_device_capability(
            torch.cuda.current_device()
        ) >= (9, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert is_kernel_allowed(Kernel.scattermoe)

        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = x.shape

        x = x.view(-1, self.hidden_size)

        x = tensor_to_dtensor(x, device_mesh=self.tp_mesh, current_placement=self.placement)

        router_logits, router_weights, selected_experts = self._compute_routing_weights(x)

        x = dtensor_to_tensor(x, device_mesh=self.tp_mesh, desired_placement=Replicate(), grad_placement=Partial())

        moe_output, expert_frequency = self._compute_experts(x, router_weights, selected_experts)

        if self.shared_intermediate_size is None:
            x = moe_output
        else:
            x = moe_output + self._compute_shared_experts(x)

        del moe_output

        x = tensor_to_dtensor(x, device_mesh=self.tp_mesh, current_placement=Partial())
        x = dtensor_to_tensor(
            x, device_mesh=self.tp_mesh, desired_placement=self.placement, grad_placement=self.placement
        )

        if not self.use_padding_free_transformer:
            x = x.reshape(batch_size, sequence_length, self.hidden_size)

        x = self.dropout(x)

        aux_loss = (
            self._compute_switch_loss(
                logits=router_logits, probs=torch.softmax(router_logits, dim=-1), expert_frequency=expert_frequency
            )
            if self.training
            else 0
        )

        add_aux_loss(aux_loss)

        return x

    def _compute_routing_weights(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x -> (total_q, hidden_size)
        router_logits = self.gate(x)
        router_logits = dtensor_to_tensor(
            router_logits, device_mesh=self.tp_mesh, desired_placement=Replicate(), grad_placement=Partial()
        )
        # router_logits -> (total_q, num_experts)

        if self.normalized_topk:
            router_weights, selected_experts = self._get_topk(router_logits)
            router_weights = F.softmax(router_weights.float(), dim=-1)
            router_weights = router_weights.type_as(x)
        else:
            router_weights = F.softmax(router_logits.float(), dim=-1)
            router_weights = router_weights.type_as(x)
            router_weights, selected_experts = self._get_topk(router_weights)

        return router_logits, router_weights, selected_experts
