# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce
from torch.utils.checkpoint import checkpoint

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import ProcessGroupManager, is_cute_kernels_available
from ...loss import add_aux_loss
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import get_activation_function, is_glu
from ..linear import ParameterizedLinear
from .mlp import _get_std_for_linear


if is_cute_kernels_available():
    from cute_kernels import continuous_count_cute
    from cute_kernels.modules.moe import (
        group_with_padding,
        grouped_gemm_experts_cute,
        scattered_experts,
        ungroup_with_padding,
    )


# TODO add support for combileable bincount in PyTorch directly
@torch.library.custom_op("lm_engine::bincount", mutates_args={})
def bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return x.bincount(minlength=minlength).to(torch.uint32)


@bincount.register_fake
def _(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return torch.empty(minlength, device=x.device, dtype=torch.uint32)


def compute_bincount(x: torch.Tensor, size: int, use_continuous_count: bool) -> torch.Tensor:
    if use_continuous_count:
        count = continuous_count_cute(x, size=size)
    else:
        count = bincount(x, minlength=size)

    return count


class ParameterizedExperts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> ParameterizedExperts:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features, device=device, dtype=dtype))

        self.bias = None
        if add_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features, device=device, dtype=dtype))

        self.std = std

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "N_array", torch.empty((num_experts,), device=device, dtype=torch.uint32), persistent=False
        )

        self.register_buffer(
            "K_array", torch.empty((num_experts,), device=device, dtype=torch.uint32), persistent=False
        )

        self.reset_parameters()

        mark_parameter_as_no_weight_decay(self.bias)

    def forward(
        self,
        input: torch.Tensor,
        num_experts_per_token: int | None = None,
        expert_frequency: torch.Tensor | None = None,
        sorted_expert_idxs: torch.Tensor | None = None,
        sorted_scattered_idxs: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
        gates: torch.Tensor | None = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        if is_kernel_allowed(Kernel.grouped_gemm_cute):
            assert self.bias is None
            assert num_experts_per_token is None
            assert sorted_expert_idxs is None
            assert sorted_scattered_idxs is None
            assert expert_offsets is None
            assert gates is None
            assert not grouped_in
            assert not grouped_out

            input = grouped_gemm_experts_cute(
                x=input, weight=self.weight, M_array=expert_frequency, N_array=self.N_array, K_array=self.K_array
            )
        elif is_kernel_allowed(Kernel.scattermoe):
            assert self.bias is None

            input = scattered_experts(
                inputs=input,
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
            input = input.split(expert_frequency.tolist(), dim=0)
            input = [
                F.linear(input[i], self.weight[i], None if self.bias is None else self.bias[i])
                for i in range(self.num_experts)
            ]
            input = torch.cat(input, dim=0)

        return input

    def extra_repr(self) -> str:
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
    linear_class = ParameterizedExperts

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
    ) -> MoE:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.shared_intermediate_size = shared_intermediate_size

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.gate = ParameterizedLinear(
            in_features=self.hidden_size,
            out_features=num_experts,
            bias=False,
            std=std,
        )

        self.c_fc = self.linear_class(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu(activation_function) else self.intermediate_size,
            add_bias=add_bias,
            std=std,
        )
        if self.shared_intermediate_size is not None:
            self.c_fc_shared = ParameterizedLinear(
                in_features=self.hidden_size,
                out_features=(
                    2 * self.shared_intermediate_size if is_glu(activation_function) else self.shared_intermediate_size
                ),
                bias=add_bias,
                std=std,
            )

        self.act = get_activation_function(activation_function)

        std /= math.sqrt(2 * num_layers)

        self.c_proj = self.linear_class(
            num_experts=num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )
        if self.shared_intermediate_size is not None:
            self.c_proj_shared = ParameterizedLinear(
                in_features=self.shared_intermediate_size,
                out_features=self.hidden_size,
                bias=add_bias,
                std=std,
            )

        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

        self.is_hopper_or_newer_gpu = torch.cuda.is_available() and torch.cuda.get_device_capability(
            torch.cuda.current_device()
        ) >= (9, 0)

        mark_parameter_as_mup_learning_rate(self.gate.weight)
        mark_parameter_as_mup_learning_rate(self.c_fc.weight)
        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

        if shared_intermediate_size is not None:
            mark_parameter_as_mup_learning_rate(self.c_fc_shared.weight)
            mark_parameter_as_mup_learning_rate(self.c_proj_shared.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)
        moe_output, expert_frequency = self._compute_experts(hidden_states, router_weights, selected_experts)

        if self.shared_intermediate_size is None:
            hidden_states = moe_output
        else:
            hidden_states = moe_output + self._compute_shared_experts(hidden_states)

        del moe_output

        hidden_states = self.dropout(hidden_states)

        aux_loss = (
            self._compute_switch_loss(
                logits=router_logits, probs=torch.softmax(router_logits, dim=-1), expert_frequency=expert_frequency
            )
            if self.training
            else 0
        )

        add_aux_loss(aux_loss)

        return hidden_states

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.gate(hidden_states)
        # router_logits -> (total_q, num_experts)

        router_weights, selected_experts = self._get_topk(router_logits)
        router_weights = F.softmax(router_weights.float(), dim=-1)

        # we cast back to the input dtype
        router_weights = router_weights.type_as(hidden_states)

        return router_logits, router_weights, selected_experts

    def _compute_experts(
        self, hidden_states: torch.Tensor, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()

            expert_frequency = compute_bincount(
                x=sorted_expert_idxs,
                size=self.num_experts,
                use_continuous_count=self.is_hopper_or_newer_gpu and is_kernel_allowed(Kernel.continuous_count_cute),
            )

        T = hidden_states.size(0)

        if is_kernel_allowed(Kernel.grouped_gemm_cute):

            def _input_projection(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                x, padded_expert_frequency, expert_padding_offset = group_with_padding(
                    x=x,
                    expert_frequency=expert_frequency,
                    sorted_idxs=sorted_expert_idxs,
                    scattered_idxs=sorted_scattered_idxs,
                    top_k=self.top_k,
                    pad_to_multiple_of=8,
                )

                x = self.c_fc(input=x, expert_frequency=padded_expert_frequency)

                return x, padded_expert_frequency, expert_padding_offset

            def _output_projection(x: torch.Tensor, padded_expert_frequency: torch.Tensor) -> torch.Tensor:
                x = self.act(x)
                x = self.c_proj(input=x, expert_frequency=padded_expert_frequency)
                return x

            if is_kernel_allowed(Kernel.checkpointed_mlp):
                hidden_states, padded_expert_frequency, expert_padding_offset = checkpoint(
                    _input_projection, hidden_states, use_reentrant=False
                )

                hidden_states = checkpoint(
                    _output_projection, hidden_states, padded_expert_frequency, use_reentrant=False
                )
            else:
                hidden_states, padded_expert_frequency, expert_padding_offset = _input_projection(hidden_states)
                hidden_states = _output_projection(hidden_states, padded_expert_frequency)

            hidden_states = ungroup_with_padding(
                x=hidden_states,
                expert_padding_offset=expert_padding_offset,
                sorted_idxs=sorted_expert_idxs,
                scattered_idxs=sorted_scattered_idxs,
                top_k=self.top_k,
                num_tokens=T,
                pad_to_multiple_of=8,
            )

            hidden_states = torch.bmm(router_weights.unsqueeze(1), hidden_states)
            hidden_states = hidden_states.squeeze(1)
        elif is_kernel_allowed(Kernel.scattermoe):
            with torch.no_grad():
                expert_offsets = expert_frequency.cumsum(-1)

            hidden_states = self.c_fc(
                input=hidden_states,
                num_experts_per_token=self.top_k,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
                grouped_out=True,
            )

            def _output_projection(x: torch.Tensor) -> torch.Tensor:
                x = self.act(x)
                x = self.c_proj(
                    input=x,
                    num_experts_per_token=1,
                    sorted_expert_idxs=sorted_expert_idxs,
                    sorted_scattered_idxs=sorted_scattered_idxs,
                    expert_offsets=expert_offsets,
                    grouped_in=True,
                    gates=router_weights,
                )

                return x

            if is_kernel_allowed(Kernel.checkpointed_mlp):
                hidden_states = checkpoint(_output_projection, hidden_states, use_reentrant=False)
            else:
                hidden_states = _output_projection(hidden_states)

            hidden_states = self.dropout(hidden_states)
        else:
            batch_index = sorted_scattered_idxs // self.top_k

            # gather the gate values for grouped input tokens
            router_weights = router_weights.flatten()
            batch_gates = router_weights[sorted_scattered_idxs]

            hidden_states = hidden_states[batch_index]

            hidden_states = self.c_fc(input=hidden_states, expert_frequency=expert_frequency)
            hidden_states = self.act(hidden_states)
            hidden_states = self.c_proj(input=hidden_states, expert_frequency=expert_frequency)

            hidden_states = hidden_states * batch_gates.unsqueeze(-1)  # [:, None]
            zeros = torch.zeros((T, self.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = zeros.index_add(0, batch_index, hidden_states)

        return hidden_states, expert_frequency

    def _compute_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc_shared(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj_shared(hidden_states)
        return hidden_states

    def _compute_expert_assignment(
        self, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_experts = selected_experts.flatten()

        num_tokens_per_expert = compute_bincount(
            x=selected_experts,
            size=self.num_experts,
            use_continuous_count=self.is_hopper_or_newer_gpu and is_kernel_allowed(Kernel.continuous_count_cute),
        )

        # sort and group input tokens according to expert assignment
        _, index_sorted_experts = selected_experts.sort(0)  # [num_tokens * top_k]
        batch_index = index_sorted_experts // self.top_k  # [num_tokens * top_k]

        # gather the gate values for grouped input tokens
        router_weights = router_weights.flatten()  # [num_tokens * top_k]
        batch_gates = router_weights[index_sorted_experts]  # [num_tokens * top_k]

        return batch_index, batch_gates, num_tokens_per_expert

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
