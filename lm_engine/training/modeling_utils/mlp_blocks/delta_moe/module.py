# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed.tensor import Partial, Replicate, Shard

from .....utils import is_sonicmoe_available
from ....dtensors import dtensor_to_tensor, tensor_to_dtensor
from ....enums import Kernel
from ....generation_cache import GenerationCache
from ....kernels import is_kernel_allowed
from ....loss import add_aux_loss
from ....parallel import ProcessGroupManager
from ....parameter import mark_parameter_as_mup_learning_rate
from ...activations import get_activation_function, is_glu
from ...attention_mask_info import AttentionMaskInfo
from ...dropout import Dropout
from ...dtensor_module import DTensorModule
from ...init_utils import _get_std_for_linear
from ...linear import ReplicatedLinear
from ...position_embedding import PositionInfo
from ..delta_mlp import DeltaMLP
from ..moe import MoE
from ..moe.experts import ColumnParallelExperts, RowParallelExperts
from .config import DeltaMoEArgs


if is_sonicmoe_available():
    from sonicmoe import moe_TC_softmax_topk_layer


class DeltaMoE(DTensorModule):
    def __init__(
        self,
        hidden_size: int,
        config: DeltaMoEArgs,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        use_depth_scaled_init: bool,
        norm_eps: float,
        layer_idx: int,
        use_padding_free_transformer: bool,
        sequence_parallel: bool = False,
    ) -> DeltaMoE:
        super().__init__()

        assert isinstance(config, DeltaMoEArgs)
        assert not ProcessGroupManager.is_tensor_parallel_enabled()

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.use_padding_free_transformer = use_padding_free_transformer
        self.hidden_size = hidden_size
        self.intermediate_size = config.intermediate_size
        self.normalized_topk = config.normalized_topk

        up_std = _get_std_for_linear(
            initializer_range=initializer_range,
            init_method=init_method,
            m_width=m_width,
            fan_in=self.hidden_size,
            num_layers=num_layers,
            use_depth_scaled_init=False,
        )

        self.gate = ReplicatedLinear(
            in_features=self.hidden_size, out_features=self.num_experts, bias=False, std=up_std
        )

        self.is_glu = is_glu(config.activation_function)

        self.c_fc = ColumnParallelExperts(
            num_experts=self.num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if self.is_glu else self.intermediate_size,
            add_bias=config.add_bias,
            std=up_std,
        )

        self.activation_function_string = config.activation_function
        self.act = get_activation_function(config.activation_function)

        self.c_proj = RowParallelExperts(
            num_experts=self.num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=config.add_bias,
            std=_get_std_for_linear(
                initializer_range=initializer_range,
                init_method=init_method,
                m_width=m_width,
                fan_in=self.intermediate_size,
                num_layers=num_layers,
                use_depth_scaled_init=use_depth_scaled_init,
            ),
        )

        self.dropout = Dropout(config.dropout)
        self.placement = Shard(0) if sequence_parallel else Replicate()

        self.delta_mlp = (
            None
            if config.delta_mlp is None
            else DeltaMLP(
                hidden_size=hidden_size,
                config=config.delta_mlp,
                layer_idx=layer_idx,
                norm_eps=norm_eps,
                init_method=init_method,
                initializer_range=initializer_range,
                m_width=m_width,
                num_layers=num_layers,
                use_depth_scaled_init=use_depth_scaled_init,
                use_padding_free_transformer=use_padding_free_transformer,
                sequence_parallel=sequence_parallel,
            )
        )

        self.is_hopper_or_newer_gpu = torch.cuda.is_available() and torch.cuda.get_device_capability(
            torch.cuda.current_device()
        ) >= (9, 0)

        self.stream_id = torch.cuda.current_stream().stream_id if torch.cuda.is_available() else None

        mark_parameter_as_mup_learning_rate(self.gate.weight)
        mark_parameter_as_mup_learning_rate(self.c_fc.weight)
        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask_info: AttentionMaskInfo | None = None,
        position_info: PositionInfo | None = None,
    ) -> torch.Tensor:
        delta_mlp_output = (
            None
            if self.delta_mlp is None
            else self.delta_mlp(
                x,
                cache_params=cache_params,
                attention_mask_info=attention_mask_info,
                position_info=position_info,
            )
        )

        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = x.shape

        x = x.view(-1, self.hidden_size)

        if self.is_tp_enabled:
            x = tensor_to_dtensor(x, device_mesh=self.tp_mesh, current_placement=self.placement)

        if is_kernel_allowed(Kernel.sonicmoe):
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
            router_logits, router_weights, selected_experts = self._compute_routing_weights(x)

            if self.is_tp_enabled:
                x = dtensor_to_tensor(
                    x, device_mesh=self.tp_mesh, desired_placement=Replicate(), grad_placement=Partial()
                )

            moe_output, expert_frequency = self._compute_experts(x, router_weights, selected_experts)

        x = moe_output
        if delta_mlp_output is not None:
            x = x + delta_mlp_output.flatten(0, 1)

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
        return MoE._compute_routing_weights(self, x=x)

    def _compute_experts(
        self, x: torch.Tensor, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return MoE._compute_experts(self, x=x, router_weights=router_weights, selected_experts=selected_experts)

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return MoE._get_topk(self, x=x)

    def _compute_switch_loss(
        self, logits: torch.Tensor, probs: torch.Tensor, expert_frequency: torch.Tensor
    ) -> torch.Tensor:
        return MoE._compute_switch_loss(self, logits=logits, probs=probs, expert_frequency=expert_frequency)

    def get_num_active_parameters(self) -> int:
        return MoE.get_num_active_parameters(self)
