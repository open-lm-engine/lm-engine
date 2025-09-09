# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_fma_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay
from ..activations import is_glu
from ..convolution import ParameterizedConv1d
from ..linear import ParameterizedLinear, ParameterizedLowRankLinear
from ..normalization import get_normalization_function
from .causal_convolution import causal_convolution
from .utils import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence, unpack_sequence


if is_fma_available():
    from fma import KernelBackend
    from fma.modules.msu import msu


class MSU(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int,
        low_rank: int | None,
        low_rank_norm: bool,
        num_heads: int,
        num_groups: int | None,
        kernel_size: int | None,
        activation_function: str | None,
        add_bias: bool,
        gradient_clipping: float | None,
        initializer_range: float,
        m_width: float,
        init_method: str,
        normalization_function: str | None,
        num_layers: int,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> MSU:
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.low_rank = low_rank
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.activation_string = activation_function
        self.gradient_clipping = gradient_clipping
        self.layer_idx = layer_idx
        self.use_padding_free_transformer = use_padding_free_transformer
        self.state_head_dim = divide_if_divisible(self.state_size, self.num_heads, "")
        self.is_gated_normalization = normalization_function == "silu_gated_rmsnorm"

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.state_weight_std = std

        self.num_q_heads = self.num_heads
        self.num_k_heads = self.num_v_heads = self.num_heads # TODO k and v are paired so must be same (?)
        self.q_state_head_dim = self.k_state_head_dim = self.state_head_dim
        self.v_state_head_dim = 2 * self.state_head_dim

        self.q_heads = self.num_q_heads * self.q_state_head_dim
        self.k_heads = self.num_k_heads * self.k_state_head_dim
        self.v_heads = self.num_v_heads * self.v_state_head_dim
        self.f_heads = self.num_k_heads * self.k_state_head_dim

        self.total_intermediate_size = self.q_heads + self.k_heads + self.v_heads + self.f_heads
        self.input_projection = ParameterizedLinear(
            self.input_size,
            (
                self.total_intermediate_size +
                (self.num_q_heads * self.v_state_head_dim if self.is_gated_normalization else 0) # residual
            ),
            bias=add_bias,
            std=std,
        )

        if kernel_size is None:
            assert num_groups is None
            assert activation_function is None
        else:
            is_glu_activation = self.activation_string is not None and is_glu(self.activation_string)
            # divide_if_divisible((6 if is_glu_activation else 3) * self.state_size, num_groups, "")

            self.conv1d = ParameterizedConv1d(
                in_channels=self.total_intermediate_size,
                out_channels=self.total_intermediate_size, # * (2 if is_glu_activation else 1),
                # out_channels=(4 if is_glu_activation else 2) * self.state_size
                # + (2 if is_glu_activation else 1) * self.num_heads,
                kernel_size=kernel_size,
                bias=add_bias,
                padding=kernel_size - 1,
                groups=self.total_intermediate_size,
                std=std,
            )

        p = torch.rand(size=(self.num_k_heads,))
        temp_logistic_p = torch.log(p) - torch.log(1 - p)
        self.log_forget_inv_temp = nn.Parameter(torch.zeros((self.num_k_heads, self.k_state_head_dim)) + temp_logistic_p[:, None])
        p = torch.linspace(0.0001, 0.9999, self.k_state_head_dim)
        logistic_p = torch.log(p) - torch.log(1 - p)
        self.forget_bias = nn.Parameter(logistic_p[None, :] / torch.sigmoid(temp_logistic_p)[:, None])

        mark_parameter_as_no_weight_decay(self.forget_bias)
        mark_parameter_as_no_weight_decay(self.log_forget_inv_temp)


        init_factor = 1 / math.sqrt(2 * self.v_state_head_dim)
        init_log_factor = math.log(math.exp(init_factor) - 1)
        self.log_factor = nn.Parameter(torch.full((self.num_v_heads,), fill_value=init_log_factor))
        mark_parameter_as_no_weight_decay(self.log_factor)
        # self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))
        self.state_weight = nn.Parameter(
            torch.cat(
                (
                    torch.eye(self.v_state_head_dim)[None, :, :] / F.softplus(self.log_factor.data)[:, None, None],
                ),
                dim=0
            )
        )
        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.norm = get_normalization_function(normalization_function, self.num_q_heads * self.v_state_head_dim)
        self.output_projection = ParameterizedLinear(self.num_q_heads * self.v_state_head_dim, self.output_size, bias=False, std=std)
        # self.q_norm = nn.GroupNorm(self.num_q_heads, self.num_q_heads * self.q_state_head_dim)
        # mark_parameter_as_no_weight_decay(self.q_norm.weight)
        # mark_parameter_as_no_weight_decay(self.q_norm.bias)
        # self.k_norm = nn.GroupNorm(self.num_k_heads, self.num_k_heads * self.k_state_head_dim)
        # mark_parameter_as_no_weight_decay(self.k_norm.weight)
        # mark_parameter_as_no_weight_decay(self.k_norm.bias)
        # self.input_norm = nn.GroupNorm(self.num_v_heads, self.num_v_heads * self.v_state_head_dim)
        # mark_parameter_as_no_weight_decay(self.input_norm.weight)
        # mark_parameter_as_no_weight_decay(self.input_norm.bias)

        self.q_norm = get_normalization_function("rmsnorm", self.q_state_head_dim, elementwise_affine=False)
        self.k_norm = get_normalization_function("rmsnorm", self.k_state_head_dim, elementwise_affine=False)
        self.input_norm = get_normalization_function("rmsnorm", self.v_state_head_dim, elementwise_affine=False)


        # self.input_norm = get_normalization_function("rmsnorm", self.state_size)
        # self.forget_norm = get_normalization_function("rmsnorm", self.state_size)
        # self.reset_norm = get_normalization_function("rmsnorm", self.num_heads)

        # self.state_weight_norm = get_normalization_function(
        #     "p_norm", self.v_state_head_dim * self.v_state_head_dim, elementwise_affine=False, p=2
        # )
        mark_parameter_as_mup_learning_rate(self.conv1d.weight)

        if self.low_rank is None:
            mark_parameter_as_mup_learning_rate(self.input_projection.weight)
        else:
            mark_parameter_as_mup_learning_rate(self.input_projection.l1.weight)
            mark_parameter_as_mup_learning_rate(self.input_projection.l2.weight)

        mark_parameter_as_mup_learning_rate(self.state_weight)
        mark_parameter_as_mup_learning_rate(self.output_projection.weight)

        mark_parameter_as_no_weight_decay(self.state_weight)

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if self.use_padding_free_transformer:
            assert cache_params is None
            assert attention_mask is None
        else:
            assert cu_seqlens is None
            assert max_seqlen is None

            B, S = input.size()[:2]

            if attention_mask is not None:
                cu_seqlens, max_seqlen = compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask)
                input = pack_sequence(inputs=input, cu_seqlens=cu_seqlens)

        input_state = None if cache_params is None else cache_params.get_cache(self.layer_idx)
        conv_state = None

        if self.low_rank is None:
            input = self.input_projection(input)

            if self.is_gated_normalization:
                input, gate = input.split((self.total_intermediate_size,
                                           self.num_q_heads * self.v_state_head_dim), dim=-1)
        else:
            if self.is_gated_normalization:
                gate = self.gate_projection(input)
            forget_input = self.forget_projection(input)
            reset_input = self.reset_projection(input)
            input = self.input_projection(input)

            input = torch.cat([input, forget_input, reset_input], dim=-1)

        if self.kernel_size is not None:
            input, conv_state = causal_convolution(
                hidden_states=input,
                input_state=conv_state,
                attention_mask=attention_mask,
                conv1d_weight=self.conv1d.weight,
                conv1d_bias=self.conv1d.bias,
                conv1d_num_groups=self.total_intermediate_size, # self.num_groups,
                return_cache_state=cache_params is not None,
                activation_string=self.activation_string,
                conv1d_padding=self.kernel_size - 1,
                conv1d_stride=1,
            )

        # input, forget_input, reset_input = input.split((self.state_size, self.state_size, self.num_heads), dim=-1)
        q_heads, k_heads, v_heads, f_heads = input.split((
            self.q_heads, self.k_heads, self.v_heads,
            self.f_heads,
        ), dim=-1)

        factor = F.softplus(self.log_factor) 
        # q, k, v: B, S, num_{q,k,v}_heads, state_head_dim
        q_heads_size = q_heads.size()
        # q_heads = q_heads.view(-1, self.num_q_heads * self.q_state_head_dim)
        q_heads = q_heads.view(*q_heads_size[:-1], self.num_q_heads, self.q_state_head_dim)
        q_heads = self.q_norm(q_heads)

        k_heads_size = k_heads.size()
        # k_heads = k_heads.view(-1, self.num_k_heads * self.k_state_head_dim)
        k_heads = k_heads.view(*k_heads_size[:-1], self.num_k_heads, self.k_state_head_dim)
        k_heads = self.k_norm(k_heads)

        v_heads_size = v_heads.size()
        # v_heads = v_heads.view(-1, self.num_v_heads * self.v_state_head_dim)
        v_heads = v_heads.view(*v_heads_size[:-1], self.num_v_heads, self.v_state_head_dim)
        v_heads = self.input_norm(v_heads)
        v_heads = v_heads * factor[:, None]

        f_heads_size = f_heads.size()
        f_heads = f_heads.view(*f_heads_size[:-1], self.num_k_heads, self.k_state_head_dim)
        f_heads = 2 * torch.sigmoid(self.log_forget_inv_temp) * (f_heads + self.forget_bias)

        kvT = k_heads[..., :, None] * v_heads[..., None, :]
        # batch_size, length, num_heads, k_dim, v_dim
        f_heads = f_heads[..., :, None].expand_as(kvT)

        expanded_input = kvT.permute(0, 3, 1, 2, 4).flatten(0, 1)
        forget_input = f_heads.permute(0, 3, 1, 2, 4).flatten(0, 1)
        # batch_size, k_dim, length, num_heads, v_dim

        reset_input = torch.full_like(expanded_input[..., :, 0], fill_value=20.)
        # forget_input = torch.full_like(expanded_input, fill_value=-40.)
        # forget_input = self.forget_norm(forget_input)
        # reset_input = self.reset_norm(reset_input)

        # input, forget_input = [
        #     i.view(*input.size()[:-1], self.num_heads, self.state_head_dim) for i in (input, forget_input)
        # ]

        # state_weight = self.state_weight_norm(self.state_weight.view(self.state_weight.size(0), -1)).view_as(self.state_weight)
        state_weight = self.state_weight * factor[:, None, None]
        # weight, forget_weight = state_weight.chunk(2, dim=0)
        weight = state_weight
        reset_weight = torch.zeros_like(weight)
        forget_weight = torch.zeros_like(weight)
        # weight, forget_weight, reset_weight = state_weight.chunk(3, dim=0)

        expanded_output = msu(
            input=expanded_input,
            weight=weight,
            forget_input=forget_input,
            # forget_weight=forget_weight * self.factor,

            # input=expanded_input,
            # weight=weight,
            # forget_input=forget_input,
            forget_weight=forget_weight,

            reset_input=reset_input,
            reset_weight=reset_weight,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.triton if is_kernel_allowed(Kernel.gru) else KernelBackend.torch,
        )

        expanded_output = expanded_output \
            .view(B, self.k_state_head_dim, S, self.num_v_heads, self.v_state_head_dim) \
            .permute(0, 2, 3, 1, 4)

        # expanded_output = B, S, k_heads, state_head_dim, state_head_dim
        # q_heads = B, S, q_heads, state_head_dim
        q_heads = q_heads.view(B, S, self.num_q_heads // self.num_k_heads, self.num_k_heads, self.q_state_head_dim)
        # q_heads = B, S, num_groups, k_heads, state_head_dim
        expanded_output = expanded_output.unsqueeze(2)
        # expanded_output = B, S, 1, k_heads, state_head_dim, state_head_dim
        input = (q_heads.unsqueeze(-2) @ expanded_output).squeeze(-2)
        # input = B, S, num_groups, k_heads, state_head_dim
        input = input.flatten(2, 3)
        # input = B, S, q_heads, state_head_dim

        # print(q_heads.size(), expanded_output.size(), input.size())
        if not self.use_padding_free_transformer and attention_mask is not None:
            input = unpack_sequence(inputs=input, cu_seqlens=cu_seqlens, output_shape=(B, S, *input.size()[1:]))

        if cache_params is not None:
            cache_params.update(state=input[:, -1], num_tokens_added=input.size(1), layer_idx=self.layer_idx)

        input = input.view(*input.size()[:-2], -1)

        if self.is_gated_normalization:
            input = self.norm(input, gate)
        else:
            input = self.norm(input)

        input = self.output_projection(input)

        return input

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # self.state_weight.data[:] = 0.
        # self.state_weight.data[None, :, :] = torch.eye(self.state_weight.size(-1),
        #                                                dtype=self.state_weight.dtype,
        #                                                device=self.state_weight.device)
        # nn.init.normal_(self.state_weight, std=self.state_weight_std)
        pass

    def extra_repr(self) -> str:
        return f"gradient_clipping = {self.gradient_clipping}\nweight_shape: {str(self.state_weight.shape)}"
