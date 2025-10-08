# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import is_fma_available
from ...cache import GenerationCache
from .causal_convolution import causal_convolution
from .fru import FRU
from .utils import compute_cu_seqlens_and_max_seqlen_from_attention_mask, pack_sequence


if is_fma_available():
    from fma import KernelBackend
    from fma.modules.rsa import rsa


class RSA(FRU):
    def forward(
        self,
        input: torch.Tensor,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        conv_state, rsa_state = (None, None) if cache_params is None else cache_params.get_cache(self.layer_idx)

        input = self.input_projection(input)
        input, gate = input.split((self.conv_dim, self.g_shape), dim=-1)

        if self.kernel_size is not None:
            input, conv_state = causal_convolution(
                hidden_states=input,
                input_state=conv_state,
                attention_mask=attention_mask,
                conv1d_weight=self.conv1d.weight,
                conv1d_bias=self.conv1d.bias,
                conv1d_num_groups=self.conv_dim,
                return_cache_state=cache_params is not None,
                activation_string=self.activation_string,
                conv1d_padding=self.kernel_size - 1,
                conv1d_stride=1,
            )

        q, k, v, f = input.split((self.q_shape, self.k_shape, self.v_shape, self.f_shape), dim=-1)

        q = q.view(*q.size()[:-1], self.num_q_heads, self.qk_head_dim)
        k = k.view(*k.size()[:-1], self.num_k_heads, self.qk_head_dim)
        v = v.view(*v.size()[:-1], self.num_v_heads, self.v_head_dim)
        f = f.view(*f.size()[:-1], self.num_f_heads, self.qk_head_dim)

        k = self.norm(k)
        f = (2 * torch.sigmoid(self.forget_multiplier[..., None])) * (f + self.forget_bias)

        input, rsa_state = rsa(
            query=q,
            key=k,
            value=v,
            weight=self.state_weight,
            forget_input=f,
            input_state=rsa_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.triton if is_kernel_allowed(Kernel.gru) else KernelBackend.torch,
        )

        if cache_params is not None:
            cache_params.update(
                conv_state=conv_state, ssm_state=rsa_state, num_tokens_added=input.size(1), layer_idx=self.layer_idx
            )

        input = input.flatten(-2, -1)
        input = input * F.silu(gate)
        input = self.g_norm(input)

        input = self.output_projection(input)

        return input
