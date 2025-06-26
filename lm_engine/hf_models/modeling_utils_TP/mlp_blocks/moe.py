# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard

from ....dtensors import dtensor_to_tensor, tensor_to_dtensor
from ....enums import Kernel
from ....kernels import is_kernel_allowed, wait_for_ACT
from ....utils import ProcessGroupManager, divide_if_divisible, is_cute_kernels_available
from ...loss import add_aux_loss
from ...modeling_utils import MoE, ParameterizedExperts, ParameterizedLinear, get_activation_function, is_glu
from ...modeling_utils.mlp_blocks.mlp import _get_std_for_linear
from ..dtensor_module import DTensorModule
from ..linear import ColumnParallelLinear, RowParallelLinear


if is_cute_kernels_available():
    from cute_kernels.modules.moe import scattered_experts


class ReplicatedLinear_TP(ParameterizedLinear, DTensorModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype, std=std
        )

        self.weight = nn.Parameter(
            tensor_to_dtensor(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Replicate()
            )
        )


class ColumnParallelExperts(ParameterizedExperts, DTensorModule):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        self.out_features_per_device = divide_if_divisible(
            out_features,
            tp_world_size,
            f"`out_features` ({out_features}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
        )

        super().__init__(
            num_experts=num_experts,
            in_features=in_features,
            out_features=self.out_features_per_device,
            add_bias=add_bias,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.weight = nn.Parameter(
            tensor_to_dtensor(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Shard(1)
            )
        )

    def forward(
        self,
        input: torch.Tensor,
        num_experts_per_token: int | None = None,
        num_tokens_per_expert: torch.Tensor | None = None,
        sorted_expert_idxs: torch.Tensor | None = None,
        sorted_scattered_idxs: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
        gates: torch.Tensor | None = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        assert is_kernel_allowed(Kernel.scattermoe)

        input = scattered_experts(
            inputs=wait_for_ACT(input, wait_in_forward=True, wait_in_backward=False),
            expert_weights=dtensor_to_tensor(self.weight).permute(0, 2, 1),
            k=num_experts_per_token,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            expert_offsets=expert_offsets,
            gates=gates,
            grouped_in=grouped_in,
            grouped_out=grouped_out,
        )

        input = wait_for_ACT(input, wait_in_forward=False, wait_in_backward=True)

        return input


class RowParallelExperts(ColumnParallelExperts):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        self.in_features_per_device = divide_if_divisible(
            in_features,
            tp_world_size,
            f"`in_features` ({in_features}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
        )

        ParameterizedExperts.__init__(
            self,
            num_experts=num_experts,
            in_features=self.in_features_per_device,
            out_features=out_features,
            add_bias=add_bias,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.weight = nn.Parameter(
            tensor_to_dtensor(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Shard(-1)
            )
        )


class SharedExpertsColumnParallelLinear(ColumnParallelLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, dtensor_to_tensor(self.weight), dtensor_to_tensor(self.bias))


class SharedExpertsRowParallelLinear(RowParallelLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, dtensor_to_tensor(self.weight), dtensor_to_tensor(self.bias))


class MoE_TP(MoE, DTensorModule):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        use_padding_free_transformer: bool,
        sequence_parallel: bool = False,
    ) -> None:
        nn.Module.__init__(self)

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.use_padding_free_transformer = use_padding_free_transformer

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.shared_intermediate_size = shared_intermediate_size

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
            add_bias=add_bias,
            std=std,
        )
        if self.shared_intermediate_size is not None:
            self.c_fc_shared = SharedExpertsColumnParallelLinear(
                in_features=self.hidden_size,
                out_features=(
                    2 * self.shared_intermediate_size if is_glu(activation_function) else self.shared_intermediate_size
                ),
                bias=add_bias,
                std=std,
            )

        self.act = get_activation_function(activation_function)

        std /= math.sqrt(2 * num_layers)

        self.c_proj = RowParallelExperts(
            num_experts=num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )
        if self.shared_intermediate_size is not None:
            self.c_proj_shared = SharedExpertsRowParallelLinear(
                in_features=self.shared_intermediate_size,
                out_features=self.hidden_size,
                bias=add_bias,
                std=std,
            )

        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.placement = Shard(0) if sequence_parallel else Replicate()

        self.is_hopper_or_newer_gpu = torch.cuda.is_available() and torch.cuda.get_device_capability(
            torch.cuda.current_device()
        ) >= (9, 0)

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.gate(hidden_states)
        router_logits = dtensor_to_tensor(
            router_logits, device_mesh=self.tp_mesh, desired_placement=Replicate(), grad_placement=Partial()
        )
        # router_logits -> (total_q, num_experts)

        router_weights, selected_experts = self._get_topk(router_logits)
        router_weights = F.softmax(router_weights.float(), dim=-1)

        # we cast back to the input dtype
        router_weights = router_weights.type_as(hidden_states)

        return router_logits, router_weights, selected_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert is_kernel_allowed(Kernel.scattermoe)

        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)

        hidden_states = tensor_to_dtensor(hidden_states, device_mesh=self.tp_mesh, current_placement=self.placement)

        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)

        hidden_states = dtensor_to_tensor(
            hidden_states, device_mesh=self.tp_mesh, desired_placement=Replicate(), grad_placement=Partial()
        )

        moe_output = self._compute_experts(hidden_states, router_weights, selected_experts)

        if self.shared_intermediate_size is None:
            hidden_states = moe_output
        else:
            hidden_states = moe_output + self._compute_shared_experts(hidden_states)

        del moe_output

        hidden_states = tensor_to_dtensor(hidden_states, device_mesh=self.tp_mesh, current_placement=Partial())
        hidden_states = dtensor_to_tensor(
            hidden_states, device_mesh=self.tp_mesh, desired_placement=self.placement, grad_placement=self.placement
        )

        if not self.use_padding_free_transformer:
            hidden_states = hidden_states.reshape(batch_size, sequence_length, self.hidden_size)

        hidden_states = self.dropout(hidden_states)

        aux_loss = (
            self._compute_switch_loss(
                logits=router_logits, probs=torch.softmax(router_logits, dim=-1), topk_idxs=selected_experts
            )
            if self.training
            else 0
        )

        add_aux_loss(aux_loss)

        return hidden_states
