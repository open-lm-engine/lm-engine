# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import Shard

from ....dtensors import dtensor_to_tensor, tensor_to_dtensor
from ....enums import Kernel
from ....kernels import is_kernel_allowed, wait_for_ACT
from ....parallel import ProcessGroupManager
from ....parameter import mark_parameter_as_initialized, mark_parameter_as_no_weight_decay
from ....utils import divide_if_divisible, is_triton_available, is_xma_available
from ...dtensor_module import DTensorModule
from ...linear import ColumnParallelLinear, RowParallelLinear


if is_xma_available() and is_triton_available():
    from xma import continuous_count

    if is_triton_available():
        from xma.layers.moe import down_projection_experts, up_projection_experts


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
    ) -> torch.Tensor:
        if self.is_tp_enabled:
            assert is_kernel_allowed(Kernel.scattermoe)

        weight = dtensor_to_tensor(self.weight)

        if is_kernel_allowed(Kernel.scattermoe):
            x = wait_for_ACT(x, wait_in_forward=True, wait_in_backward=False)

            x = up_projection_experts(
                x=x,
                expert_weights=weight.permute(0, 2, 1),
                k=num_experts_per_token,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
            )

            x = wait_for_ACT(x, wait_in_forward=False, wait_in_backward=True)
        else:
            x = x.split(expert_frequency.tolist(), dim=0)
            x = [F.linear(x[i], weight[i]) for i in range(self.num_experts)]
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

    def forward(
        self,
        x: torch.Tensor,
        num_experts_per_token: int | None = None,
        expert_frequency: torch.Tensor | None = None,
        sorted_expert_idxs: torch.Tensor | None = None,
        sorted_scattered_idxs: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
        router_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_tp_enabled:
            assert is_kernel_allowed(Kernel.scattermoe)

        weight = dtensor_to_tensor(self.weight)

        if is_kernel_allowed(Kernel.scattermoe):
            x = wait_for_ACT(x, wait_in_forward=True, wait_in_backward=False)

            x = down_projection_experts(
                x=x,
                expert_weights=weight.permute(0, 2, 1),
                k=num_experts_per_token,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
                router_weights=router_weights,
            )

            x = wait_for_ACT(x, wait_in_forward=False, wait_in_backward=True)
        else:
            x = x.split(expert_frequency.tolist(), dim=0)
            x = [F.linear(x[i], weight[i]) for i in range(self.num_experts)]
            x = torch.cat(x, dim=0)

        return x

    def extra_repr(self) -> str:
        return "num_experts={}, in_features_per_tp_rank={}, out_features={}".format(
            self.num_experts, self.in_features_per_tp_rank, self.out_features
        )
