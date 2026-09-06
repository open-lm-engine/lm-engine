# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...accelerator import Accelerator, KernelBackend
from ...functional import continuous_count
from ...utils import is_triton_available


if is_triton_available():
    from .triton_implementation import down_projection_experts, up_projection_experts


class Experts(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, add_bias: bool = True, std: float | None = None
    ) -> Experts:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))

        self.bias = None
        if add_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features))

        self.std = std

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("N_array", torch.empty((num_experts,), dtype=torch.uint32))
        self.register_buffer("K_array", torch.empty((num_experts,), dtype=torch.uint32))

        self.reset_parameters()

    def up_projection_triton_forward(
        self,
        x: torch.Tensor,
        num_experts_per_token: int | None = None,
        sorted_expert_idxs: torch.Tensor | None = None,
        sorted_scattered_idxs: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.bias is None

        x = up_projection_experts(
            x=x,
            expert_weights=self.weight.permute(0, 2, 1),
            k=num_experts_per_token,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            expert_offsets=expert_offsets,
        )

        return x

    def down_projection_triton_forward(
        self,
        x: torch.Tensor,
        num_experts_per_token: int | None = None,
        sorted_expert_idxs: torch.Tensor | None = None,
        sorted_scattered_idxs: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
        router_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.bias is None

        x = down_projection_experts(
            x=x,
            expert_weights=self.weight.permute(0, 2, 1),
            k=num_experts_per_token,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            expert_offsets=expert_offsets,
            router_weights=router_weights,
        )

        return x

    def torch_forward(
        self, x: torch.Tensor, expert_frequency: torch.Tensor | None, return_list: bool = False
    ) -> list[torch.Tensor] | torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = x.split(expert_frequency.tolist(), dim=0)
        else:
            assert expert_frequency is None

        x = [
            F.linear(x[i], self.weight[i], None if self.bias is None else self.bias[i])
            for i in range(self.num_experts)
        ]

        if not return_list:
            x = torch.cat(x, dim=0)

        return x

    def extra_repr(self):
        return "num_experts={}, in_features={}, out_features={}".format(
            self.num_experts, self.in_features, self.out_features
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)
        if hasattr(self, "bias") and self.bias is not None:
            self.bias.zero_()

        self.N_array.fill_(self.out_features)
        self.K_array.fill_(self.in_features)


class MoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function: Callable,
        is_glu: bool,
        add_bias: bool,
        std: float,
    ) -> MoE:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate = nn.Linear(in_features=self.hidden_size, out_features=num_experts, bias=False)

        self.c_fc = Experts(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu else self.intermediate_size,
            add_bias=add_bias,
            std=std,
        )

        self.act = activation_function

        self.c_proj = Experts(
            num_experts=num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )

    def forward(self, x: torch.Tensor, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
        original_shape = x.shape

        # x -> (batch_size, query_length, hidden_size)
        x = x.view(-1, self.hidden_size)
        # x -> (total_q, hidden_size)
        router_logits, router_weights, selected_experts = self._compute_routing_weights(x)

        # router_logits -> (total_q, num_experts)
        # router_weights -> (total_q, top_k)
        # selected_experts -> (total_q, top_k)

        x = self._compute_experts(x, router_weights, selected_experts, kernel_backend=kernel_backend)

        x = x.view(original_shape)

        # x -> (batch_size, query_length, hidden_size)

        return x, router_logits

    def _compute_routing_weights(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        # x -> (total_q, hidden_size)
        router_logits = self.gate(x)
        # router_logits -> (total_q, num_experts)

        router_weights, selected_experts = self._get_topk(router_logits)

        # router_weights -> (total_q, top_k)
        # selected_experts -> (total_q, top_k)

        router_weights = F.softmax(router_weights.float(), dim=-1)
        router_weights = router_weights.type_as(x)

        return router_logits, router_weights, selected_experts

    def _compute_experts(
        self,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        *,
        kernel_backend: KernelBackend | None = None,
    ) -> torch.Tensor:
        if kernel_backend is None:
            kernel_backend = Accelerator.get_kernel_backend()
        else:
            assert kernel_backend.verify_accelerator()

        sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()
        expert_frequency = continuous_count(sorted_expert_idxs, self.num_experts)

        T = x.size(0)

        if kernel_backend in [KernelBackend.cuda, KernelBackend.triton]:
            with torch.no_grad():
                expert_offsets = expert_frequency.cumsum(-1)

            x = self.c_fc.up_projection_triton_forward(
                x=x,
                num_experts_per_token=self.top_k,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
            )
            x = self.act(x)
            x = self.c_proj.down_projection_triton_forward(
                x=x,
                num_experts_per_token=1,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
                router_weights=router_weights,
            )
        elif kernel_backend == KernelBackend.torch:
            # sort and group input tokens according to expert assignment
            fan_in_index = sorted_scattered_idxs // self.top_k

            # gather the gate values for grouped input tokens
            router_weights = router_weights.flatten()
            batch_gates = router_weights[sorted_scattered_idxs]

            x = x[fan_in_index]
            x = self.c_fc.torch_forward(x=x, expert_frequency=expert_frequency, return_list=True)
            x = [self.act(i) for i in x]
            x = self.c_proj.torch_forward(x=x, expert_frequency=None, return_list=False)

            x = x * batch_gates.unsqueeze(-1)
            zeros = torch.zeros((T, self.hidden_size), dtype=x.dtype, device=x.device)
            x = zeros.index_add(0, fan_in_index, x)
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return x

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            x, indices = x.max(dim=-1, keepdim=True)
        else:
            x, indices = x.topk(self.top_k, dim=-1)

        return x, indices
