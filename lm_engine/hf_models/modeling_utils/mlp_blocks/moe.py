# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard

from ....dtensors import dtensor_to_tensor, tensor_to_dtensor
from ....enums import Kernel
from ....kernels import is_kernel_allowed, wait_for_ACT
from ....utils import ProcessGroupManager, divide_if_divisible, is_sonicmoe_available, is_xma_available
from ...loss import add_aux_loss
from ...parameter import (
    mark_parameter_as_initialized,
    mark_parameter_as_mup_learning_rate,
    mark_parameter_as_no_weight_decay,
)
from ..activations import get_activation_function, is_glu
from ..dropout import Dropout
from ..dtensor_module import DTensorModule
from ..linear import ColumnParallelLinear, ParameterizedLinear, ReplicatedLinear, RowParallelLinear
from .mlp import _get_std_for_linear


if is_xma_available():
    from xma import continuous_count
    from xma.layers.moe import scattered_experts


if is_sonicmoe_available():
    from sonicmoe import moe_TC_softmax_topk_layer


# TODO add support for combileable bincount in PyTorch directly
@torch.library.custom_op("lm_engine::bincount", mutates_args={})
def bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return x.bincount(minlength=minlength).to(torch.uint32)


@bincount.register_fake
def _(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return torch.empty(minlength, device=x.device, dtype=torch.uint32)


def compute_bincount(x: torch.Tensor, size: int, use_continuous_count: bool) -> torch.Tensor:
    if use_continuous_count:
        count = continuous_count(x, bins=size)
    else:
        count = bincount(x, minlength=size)

    return count


class SharedExpertsColumnParallelLinear(ColumnParallelLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, dtensor_to_tensor(self.weight), dtensor_to_tensor(self.bias))


class SharedExpertsRowParallelLinear(RowParallelLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, dtensor_to_tensor(self.weight), dtensor_to_tensor(self.bias))


class ParameterizedExperts(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, add_bias: bool, std: float | None = None
    ) -> ParameterizedExperts:
        super().__init__()

        self.std = std

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))

        self.bias = None
        if add_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features))

        mark_parameter_as_no_weight_decay(self.bias)

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        num_experts_per_token: int | None = None,
        expert_frequency: torch.Tensor | None = None,
        sorted_expert_idxs: torch.Tensor | None = None,
        sorted_scattered_idxs: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
        gates: torch.Tensor | None = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        if is_kernel_allowed(Kernel.scattermoe):
            x = scattered_experts(
                inputs=x,
                expert_weights=self.weight.permute(0, 2, 1),
                k=num_experts_per_token,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
                gates=gates,
                grouped_in=grouped_in,
                grouped_out=grouped_out,
            )
        else:
            x = x.split(expert_frequency.tolist(), dim=0)
            x = [F.linear(x[i], self.weight[i]) for i in range(self.num_experts)]
            x = torch.cat(x, dim=0)

        return x

    def extra_repr(self) -> str:
        return "num_experts={}, in_features={}, out_features={}".format(
            self.num_experts, self.in_features, self.out_features
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        mark_parameter_as_initialized(self.weight)
        mark_parameter_as_initialized(self.bias)


class ColumnParallelExperts(ParameterizedExperts, DTensorModule):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, add_bias: bool, std: float | None = None
    ) -> ColumnParallelExperts:
        DTensorModule.__init__(self)

        self.out_features_per_tp_rank = divide_if_divisible(
            out_features,
            self.tp_world_size,
            f"`out_features` ({out_features}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})",
        )

        ParameterizedExperts.__init__(
            self,
            num_experts=num_experts,
            in_features=in_features,
            out_features=self.out_features_per_tp_rank,
            add_bias=add_bias,
            std=std,
        )

        if self.is_tp_enabled:
            assert not add_bias

            self.weight = nn.Parameter(
                tensor_to_dtensor(
                    self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), current_placement=Shard(1)
                )
            )

            self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        num_experts_per_token: int | None = None,
        expert_frequency: torch.Tensor | None = None,
        sorted_expert_idxs: torch.Tensor | None = None,
        sorted_scattered_idxs: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
        gates: torch.Tensor | None = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        if self.is_tp_enabled:
            assert is_kernel_allowed(Kernel.scattermoe)

        if is_kernel_allowed(Kernel.scattermoe):
            x = wait_for_ACT(x, wait_in_forward=True, wait_in_backward=False)

            x = scattered_experts(
                inputs=x,
                expert_weights=dtensor_to_tensor(self.weight).permute(0, 2, 1),
                k=num_experts_per_token,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
                gates=gates,
                grouped_in=grouped_in,
                grouped_out=grouped_out,
            )

            x = wait_for_ACT(x, wait_in_forward=False, wait_in_backward=True)
        else:
            x = x.split(expert_frequency.tolist(), dim=0)
            x = [F.linear(x[i], self.weight[i]) for i in range(self.num_experts)]
            x = torch.cat(x, dim=0)

        return x

    def extra_repr(self) -> str:
        return "num_experts={}, in_features={}, out_features_per_tp_rank={}".format(
            self.num_experts, self.in_features, self.out_features_per_tp_rank
        )


class RowParallelExperts(ParameterizedExperts, DTensorModule):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, add_bias: bool, std: float | None = None
    ) -> RowParallelExperts:
        DTensorModule.__init__(self)

        self.in_features_per_tp_rank = divide_if_divisible(
            in_features,
            self.tp_world_size,
            f"`in_features` ({in_features}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})",
        )

        ParameterizedExperts.__init__(
            self,
            num_experts=num_experts,
            in_features=self.in_features_per_tp_rank,
            out_features=out_features,
            add_bias=add_bias,
            std=std,
        )

        if self.is_tp_enabled:
            assert not add_bias

            self.weight = nn.Parameter(
                tensor_to_dtensor(
                    self.weight,
                    device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                    current_placement=Shard(-1),
                )
            )

            self.reset_parameters()

    def extra_repr(self) -> str:
        return "num_experts={}, in_features_per_tp_rank={}, out_features={}".format(
            self.num_experts, self.in_features_per_tp_rank, self.out_features
        )


class MoE(DTensorModule):
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
        add_bias: bool,
        activation_function: str,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        use_padding_free_transformer: bool,
        sequence_parallel: bool = False,
    ) -> MoE:
        super().__init__()

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

        self.gate = ReplicatedLinear(in_features=self.hidden_size, out_features=num_experts, bias=False, std=std)

        if self.shared_expert_gating:
            assert shared_intermediate_size is not None

            self.shared_expert_gate = ParameterizedLinear(
                in_features=self.hidden_size, out_features=1, bias=False, std=std
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

        self.activation_function_string = activation_function
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

        self.dropout = Dropout(dropout)
        self.placement = Shard(0) if sequence_parallel else Replicate()

        self.is_hopper_or_newer_gpu = torch.cuda.is_available() and torch.cuda.get_device_capability(
            torch.cuda.current_device()
        ) >= (9, 0)

        self.stream_id = torch.cuda.current_stream().stream_id if torch.cuda.is_available() else None

        mark_parameter_as_mup_learning_rate(self.gate.weight)
        mark_parameter_as_mup_learning_rate(self.c_fc.weight)
        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

        if shared_intermediate_size is not None:
            mark_parameter_as_mup_learning_rate(self.c_fc_shared.weight)
            mark_parameter_as_mup_learning_rate(self.c_proj_shared.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = x.shape

        x = x.view(-1, self.hidden_size)

        if self.is_tp_enabled:
            x = tensor_to_dtensor(x, device_mesh=self.tp_mesh, current_placement=self.placement)

        if is_kernel_allowed(Kernel.sonicmoe):
            assert self.use_interleaved_weights
            assert self.activation_function_string == "swiglu"
            assert not self.is_tp_enabled

            moe_output, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
                x=x,
                router_w=self.gate.weight,
                w1=self.c_fc.weight.permute(1, 2, 0),
                b1=self.c_fc.bias,
                w2=self.c_proj.weight.permute(1, 2, 0),
                b2=self.c_proj.bias,
                K=self.top_k,
                stream_id=self.stream_id,
                is_inference_mode_enabled=False,
            )
        else:
            assert not self.use_interleaved_weights

            router_logits, router_weights, selected_experts = self._compute_routing_weights(x)

            if self.is_tp_enabled:
                x = dtensor_to_tensor(
                    x, device_mesh=self.tp_mesh, desired_placement=Replicate(), grad_placement=Partial()
                )

            moe_output, expert_frequency = self._compute_experts(x, router_weights, selected_experts)

        if self.shared_intermediate_size is None:
            x = moe_output
        else:
            x = moe_output + self._compute_shared_experts(x)

        del moe_output

        if self.is_tp_enabled:
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

        if self.is_tp_enabled:
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

    def _compute_experts(
        self, x: torch.Tensor, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()

            expert_frequency = compute_bincount(
                x=sorted_expert_idxs,
                size=self.num_experts,
                use_continuous_count=self.is_hopper_or_newer_gpu and is_kernel_allowed(Kernel.continuous_count),
            )

        T = x.size(0)

        if is_kernel_allowed(Kernel.scattermoe):
            with torch.no_grad():
                expert_offsets = expert_frequency.cumsum(-1)

            x = self.c_fc(
                x=x,
                num_experts_per_token=self.top_k,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
                grouped_out=True,
            )

            x = self.act(x)

            x = self.c_proj(
                x=x,
                num_experts_per_token=1,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
                grouped_in=True,
                gates=router_weights,
            )

            x = self.dropout(x)
        else:
            batch_index = sorted_scattered_idxs // self.top_k

            # gather the gate values for grouped input tokens
            router_weights = router_weights.flatten()
            batch_gates = router_weights[sorted_scattered_idxs]

            x = x[batch_index]

            x = self.c_fc(x=x, expert_frequency=expert_frequency)
            x = self.act(x)
            x = self.c_proj(x=x, expert_frequency=expert_frequency)

            x = x * batch_gates.unsqueeze(-1)  # [:, None]
            zeros = torch.zeros((T, self.hidden_size), dtype=x.dtype, device=x.device)
            x = zeros.index_add(0, batch_index, x)

        return x, expert_frequency

    def _compute_shared_experts(self, x: torch.Tensor) -> torch.Tensor:
        g = None
        if self.shared_expert_gating:
            g = self.shared_expert_gate(x)

        x = self.c_fc_shared(x)
        x = self.act(x)
        x = self.c_proj_shared(x)

        if g is not None:
            x = x * F.sigmoid(g)

        return x

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            x, indices = x.max(dim=-1, keepdim=True)
        else:
            x, indices = x.topk(self.top_k, dim=-1)

        return x, indices

    def _compute_switch_loss(
        self, logits: torch.Tensor, probs: torch.Tensor, expert_frequency: torch.Tensor
    ) -> torch.Tensor:
        logits = logits.view(-1, logits.size(-1))
        probs = probs.view(-1, probs.size(-1))

        num_experts = logits.size(1)
        acc_probs = probs.sum(0)

        expert_frequency = expert_frequency.float()

        if ProcessGroupManager.is_initialized() and ProcessGroupManager.get_data_parallel_world_size() > 1:
            expert_frequency = all_reduce(
                expert_frequency, reduceOp="sum", group=ProcessGroupManager.get_data_parallel_group()
            )

        switch_loss = (
            num_experts * (F.normalize(acc_probs, p=1, dim=0) * F.normalize(expert_frequency, p=1, dim=0)).sum()
        )
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        loss = switch_loss + 0.1 * z_loss

        return loss.type_as(logits)

    def get_num_active_parameters(self) -> int:
        num_elements = 0
        for parameter in self.parameters():
            num_elements += parameter.numel()

        for parameter in self.c_fc.parameters():
            num_elements -= parameter.numel()
            num_elements += (parameter.numel() * self.top_k) // self.num_experts

        for parameter in self.c_proj.parameters():
            num_elements -= parameter.numel()
            num_elements += (parameter.numel() * self.top_k) // parameter.size(0)

        return num_elements
