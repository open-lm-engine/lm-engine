# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GenerationConfig, PreTrainedModel

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import Accelerator
from ...cache import GenerationCache
from ...config import CommonConfig
from ...modeling_utils import Dropout, ParameterizedEmbedding, RoPE, YaRNScaledRoPE, get_normalization_function
from ...utils import convert_padding_free_lists_to_tensors, is_generation_cache_enabled
from ..modeling_outputs import BaseModelOutputWithPast
from .layer import Block


class PreTrainedModelMixin(PreTrainedModel):
    config_class = None
    layer_class = Block
    base_model_prefix = "transformer"
    causal = True
    _no_split_modules = ["Block"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, config: CommonConfig, *args, **kwargs) -> PreTrainedModelMixin:
        super().__init__(config, *args, **kwargs)

        self.sequence_parallel = kwargs.get("sequence_parallel", False)
        self.num_pipeline_stages = kwargs.get("num_pipeline_stages", 1)
        self.pipeline_stage_id = kwargs.get("pipeline_stage_id", 0)

        self.is_first_stage = self.pipeline_stage_id == 0
        self.is_last_stage = self.pipeline_stage_id == self.num_pipeline_stages - 1
        self.is_pipeline_parallel_enabled = self.num_pipeline_stages > 1

        assert self.config_class is not None
        self.generation_config = GenerationConfig.from_model_config(self.config)

        self.use_padding_free_transformer = kwargs.get("use_padding_free_transformer", False)
        self._tied_word_embeddings = config.tie_word_embeddings

        self._has_mamba2 = any([block.sequence_mixer_type == "mamba2" for block in self.config.sequence_mixer_blocks])

        if self.is_pipeline_parallel_enabled and self._tied_word_embeddings:
            raise NotImplementedError()

    # FIXME typing
    def prepare_inputs_for_model(
        self,
        input_ids: torch.Tensor | list[list[int]] | None,
        position_ids: torch.Tensor | list[list[int]] | None,
        labels: torch.Tensor | list[list[int]] | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
        past_key_values: tuple[tuple[torch.Tensor]],
        attention_mask: torch.Tensor | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor]:
        if self.use_padding_free_transformer:
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
                    device=Accelerator.get_current_device(),
                )
            else:
                assert (
                    cu_seqlens is not None
                ), "cu_seqlens needs to be specified when using tensor inputs with padding_free transformer"
                assert position_ids is not None, "max_seqlen needs to be specified when specifying cu_seqlens"
                assert max_seqlen is not None, "max_seqlen needs to be specified when specifying cu_seqlens"
                assert attention_mask is None, "attention_mask should not be passed when specifying cu_seqlens"

            if use_cache or past_key_values is not None:
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

        self.embedding_dropout = Dropout(config.embedding_dropout)
        self.h = nn.ModuleList(
            [
                self.layer_class(
                    config,
                    use_padding_free_transformer=self.use_padding_free_transformer,
                    sequence_parallel=self.sequence_parallel,
                    layer_idx=i,
                )
                for i in range(config.num_layers)
            ]
        )
        self.ln_f = get_normalization_function(
            config.normalization_function, self.embed_dim, eps=config.layer_norm_epsilon
        )

        self.rope_dim = config.rope_dim

        self.position_embedding_type = config.position_embedding_type
        self._setup_positional_encoding()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: GenerationCache | None = None,
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
            past_key_values,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if is_generation_cache_enabled():
            past_key_values = (
                GenerationCache(self.config) if use_cache and past_key_values is None else past_key_values
            )

        mamba_mask = None
        mamba_mask_computed = False

        for sequence_mixer_type, block in zip(self.sequence_mixer_block_types, self.h):
            is_linear_layer = sequence_mixer_type in ["mamba2", "rnn", "gru"]

            if is_linear_layer and not mamba_mask_computed:
                mamba_mask = self._get_mamba_mask(attention_mask, past_key_values)
                mamba_mask_computed = True

            hidden_states = block(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=mamba_mask if is_linear_layer else causal_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

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

        causal_mask = causal_mask.unsqueeze(1)

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
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor | None, GenerationCache | None]:
        if use_cache is None:
            use_cache = False if self.use_padding_free_transformer else self.config.use_cache

        input_shape = input_ids.size()

        # special handling for padding free transformer with list inputs
        if self.use_padding_free_transformer:
            # for flash attention, there is no padding and we do packing
            # so, input_ids is of shape (s1 + s2 + ... + sb)
            batch_size = cu_seqlens.shape[0] - 1
        else:
            batch_size = input_shape[0]

        if self.use_padding_free_transformer:
            assert position_ids is not None, (
                "GPTBaseModel needs position_ids from outside when using flash attention with List[List[int]] "
                "inputs"
            )

        past_length = None
        query_length = None
        key_length = None
        if self.use_padding_free_transformer:
            key_length = max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen
        else:
            past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            query_length = input_shape[-1]
            key_length = past_length + query_length

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
            past_key_values,
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
        self, attention_mask: torch.Tensor | None, past_key_values: GenerationCache
    ) -> torch.Tensor | None:
        mamba_mask = attention_mask
        if (
            past_key_values is None
            or past_key_values.get_seq_length() > 0
            or (attention_mask is not None and torch.all(attention_mask == 1))
        ):
            mamba_mask = None

        return mamba_mask
