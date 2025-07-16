# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ...cache import GenerationCache
from ...config import CommonConfig
from ...modeling_utils import ParameterizedEmbedding, RoPE, YaRNScaledRoPE, get_normalization_function
from ...utils import convert_padding_free_lists_to_tensors, is_generation_cache_enabled
from ..modeling_outputs import BaseModelOutputWithPast
from .layer import Block


class PreTrainedModelMixin(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = None
    layer_class = Block
    base_model_prefix = "transformer"
    causal = True
    _no_split_modules = ["Block"]
    _skip_keys_device_placement = "cache_params"

    def __init__(self, config: CommonConfig, *args, **kwargs) -> PreTrainedModelMixin:
        super().__init__(config, *args, **kwargs)

        assert self.config_class is not None
        self._tied_word_embeddings = config.tie_word_embeddings

    def _init_weights(self, module: nn.Module) -> None:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    # FIXME typing
    def prepare_inputs_for_model(
        self,
        input_ids: torch.Tensor | list[list[int]] | None,
        position_ids: torch.Tensor | list[list[int]] | None,
        labels: torch.Tensor | list[list[int]] | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
        cache_params: GenerationCache,
        attention_mask: torch.Tensor | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor]:
        if isinstance(input_ids, list):
            # this is managed internally
            error_message = (
                "{variable} should not be passed for flash attention when using List[List[int]] "
                "input types attention mask logic is handled internally"
            )
            assert cu_seqlens is None, error_message.format(variable="cu_seqlens")
            assert max_seqlen is None, error_message.format(variable="max_seqlen")
            assert attention_mask is None, error_message.format(variable="attention_mask")

            input_ids, position_ids, labels, cu_seqlens, max_seqlen = convert_padding_free_lists_to_tensors(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                device=torch.cuda.current_device(),
            )
        else:
            assert (
                cu_seqlens is not None
            ), "cu_seqlens needs to be specified when using tensor inputs with padding_free transformer"
            assert position_ids is not None, "max_seqlen needs to be specified when specifying cu_seqlens"
            assert max_seqlen is not None, "max_seqlen needs to be specified when specifying cu_seqlens"
            assert attention_mask is None, "attention_mask should not be passed when specifying cu_seqlens"

        if use_cache or cache_params is not None:
            raise NotImplementedError("KV caching is not supported with padding_free transformer")

        return input_ids, position_ids, labels, cu_seqlens, max_seqlen


class BaseModelMixin(PreTrainedModelMixin):
    mask_value = None

    def __init__(self, config: CommonConfig, **kwargs) -> BaseModelMixin:
        super().__init__(config, **kwargs)
        self._init_model(config, **kwargs)

    def _init_model(self, config: CommonConfig, **kwargs) -> None:
        self.embed_dim = config.hidden_size
        self.m_emb = config.m_emb
        self.initializer_range = config.initializer_range
        self.sequence_mixer_block_types = [
            config.sequence_mixer_blocks[i].sequence_mixer_type for i in range(config.num_layers)
        ]

        self.wte = ParameterizedEmbedding(config.vocab_size, self.embed_dim, std=self.initializer_range)

        self.embedding_dropout = (
            nn.Identity() if config.embedding_dropout == 0 else nn.Dropout(config.embedding_dropout)
        )
        self.h = nn.ModuleList([self.layer_class(config, layer_idx=i) for i in range(config.num_layers)])
        self.ln_f = get_normalization_function(
            config.normalization_function, self.embed_dim, eps=config.layer_norm_epsilon
        )

        self.rope_dim = config.rope_dim

        self.position_embedding_type = config.position_embedding_type
        self._setup_positional_encoding()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ParameterizedEmbedding:
        return self.wte

    def set_input_embeddings(self, new_embeddings: ParameterizedEmbedding) -> None:
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> BaseModelOutputWithPast:
        (
            use_cache,
            hidden_states,
            causal_mask,
            position_ids,
            rope_cos_sin,
            cache_params,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            cache_params=cache_params,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if is_generation_cache_enabled():
            cache_params = GenerationCache(self.config) if use_cache and cache_params is None else cache_params

        mamba_mask = None
        mamba_mask_computed = False

        for sequence_mixer_type, block in zip(self.sequence_mixer_block_types, self.h):
            is_linear_layer = sequence_mixer_type in ["mamba2", "rnn", "gru"]

            if is_linear_layer and not mamba_mask_computed:
                mamba_mask = self._get_mamba_mask(attention_mask, cache_params)
                mamba_mask_computed = True

            hidden_states = block(
                hidden_states,
                cache_params=cache_params,
                attention_mask=mamba_mask if is_linear_layer else causal_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, cache_params=cache_params)

    def _get_position_ids(
        self, attention_mask: torch.Tensor, past_length: int, query_length: int, key_length: int, device: torch.device
    ) -> torch.Tensor:
        if attention_mask is not None and len(attention_mask.shape) == 2:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_length > 0:
                position_ids = position_ids[:, past_length:key_length:]
        else:
            position_ids = torch.arange(past_length, key_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, query_length)

        return position_ids

    def _get_rope_cos_sin(
        self, key_length: int, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.position_embedding_type == "rope":
            cos, sin = self.rope(key_length, dtype=dtype)
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
            return cos, sin

    def _prepare_causal_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        past_length = key_length - query_length

        # ==========================================================================================
        # attention_mask -> (batch_size, key_length)
        # ==========================================================================================

        if query_length > 1:
            # (query_length, key_length)
            causal_mask = torch.empty((query_length, key_length), dtype=torch.bool, device=device)
            causal_mask[:, past_length:] = torch.tril(
                torch.ones(query_length, query_length, dtype=torch.bool, device=device)
            )

            if past_length > 0:
                causal_mask[:, :past_length] = True

            # (query_length, key_length) -> (1, query_length, key_length)
            causal_mask = causal_mask.unsqueeze(0)

            if attention_mask is None:
                # (1, query_length, key_length) -> (batch_size, query_length, key_length)
                causal_mask = causal_mask.expand(batch_size, -1, -1)
            else:
                # (1, query_length, key_length) & (batch_size, 1, key_length) -> (batch_size, query_length, key_length)
                causal_mask = causal_mask & attention_mask.unsqueeze(1).to(torch.bool)
        else:
            if attention_mask is None:
                # (batch_size, query_length, key_length)
                causal_mask = torch.ones(batch_size, query_length, key_length, dtype=torch.bool, device=device)
            else:
                # (batch_size, query_length, key_length)
                causal_mask = attention_mask.unsqueeze(1).to(dtype=torch.bool, device=device)

        # ==========================================================================================
        # attention_mask -> (batch_size, query_length, key_length)
        # ==========================================================================================

        causal_mask = causal_mask.unsqueeze(1)

        # ==========================================================================================
        # attention_mask -> (batch_size, 1, query_length, key_length)
        # ==========================================================================================

        return causal_mask

    def _get_initial_hidden_state(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None) -> torch.Tensor:
        hidden_state = self.wte(input_ids)

        if self.position_embedding_type == "learned_absolute":
            hidden_state = hidden_state + self.wpe(position_ids)

        hidden_state = self.embedding_dropout(hidden_state)

        if self.m_emb is not None:
            hidden_state = hidden_state * self.m_emb

        return hidden_state

    def _prepare_a_bunch_of_stuff(
        self,
        input_ids: torch.Tensor | None = None,
        cache_params: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor | None, GenerationCache | None]:
        if use_cache is None:
            # use_cache = False if self.use_padding_free_transformer else self.config.use_cache
            use_cache = False

        batch_size = cu_seqlens.size(0) - 1

        assert position_ids is not None, (
            "GPTBaseModel needs position_ids from outside when using flash attention with List[List[int]] " "inputs"
        )

        past_length = None
        query_length = None
        key_length = max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen

        if position_ids is None:
            position_ids = self._get_position_ids(
                attention_mask, past_length, query_length, key_length, input_ids.device
            )

        hidden_states = self._get_initial_hidden_state(input_ids, position_ids)

        rope_cos_sin = self._get_rope_cos_sin(key_length, position_ids, dtype=hidden_states.dtype)

        attention_mask = self._get_maybe_causal_mask(
            attention_mask, batch_size, query_length, key_length, hidden_states.dtype, input_ids.device
        )

        return (
            use_cache,
            hidden_states,
            attention_mask,
            position_ids,
            rope_cos_sin,
            cache_params,
        )

    def _setup_positional_encoding(self) -> None:
        max_position_embeddings = self.config.max_position_embeddings

        if self.position_embedding_type == "learned_absolute":
            self.wpe = ParameterizedEmbedding(max_position_embeddings, self.embed_dim, std=self.initializer_range)
        elif self.position_embedding_type == "rope":
            if self.config.rope_scaling is None:
                self.rope = RoPE(
                    self.rope_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=self.config.rope_theta,
                )
            else:
                self.rope = YaRNScaledRoPE(
                    self.rope_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=self.config.rope_theta,
                    scale=self.config.rope_scaling["factor"],
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"],
                )
        elif self.position_embedding_type == "nope":
            pass
        else:
            raise NotImplementedError()

    def _get_mask_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        return self.mask_value

    def _get_maybe_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        query_length: int,
        key_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not (is_kernel_allowed(Kernel.flash_attention_2) or is_kernel_allowed(Kernel.flash_attention_3)):
            # we use the causal/non-causal argument of SDPA for attention in this case
            if attention_mask is not None:
                attention_mask = self._prepare_causal_attention_mask(
                    attention_mask, batch_size, query_length, key_length, device
                )

                attention_mask = torch.where(
                    attention_mask,
                    ~attention_mask,
                    self._get_mask_value(attention_mask.device, dtype),
                )

                # this is needed to prevent NaN since SDPA
                # see issue: https://github.com/pytorch/pytorch/issues/110213
                attention_mask = attention_mask * ~torch.all(
                    attention_mask == self._get_mask_value(attention_mask.device, dtype), dim=-1, keepdim=True
                )

        return attention_mask

    def _get_mamba_mask(
        self, attention_mask: torch.Tensor | None, cache_params: GenerationCache
    ) -> torch.Tensor | None:
        mamba_mask = attention_mask
        if (
            cache_params is None
            or cache_params.get_seq_length() > 0
            or (attention_mask is not None and torch.all(attention_mask == 1))
        ):
            mamba_mask = None

        return mamba_mask
